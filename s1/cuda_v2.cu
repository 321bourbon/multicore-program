#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <cuda_runtime.h>
#include "pgm_io.h"

static void check_cuda(cudaError_t result, const char *msg){
    if(result!=cudaSuccess){
        printf("CUDA Error: %s: %s\n", msg, cudaGetErrorString(result));
        exit(1);
    }
}

void gaussian_blur_5x5(const unsigned char* src, unsigned char* dst, int width, int height){
    int kernel[5][5]={
        {1,  4,  6,  4, 1},
        {4, 16, 24, 16, 4},
        {6, 24, 36, 24, 6},
        {4, 16, 24, 16, 4},
        {1,  4,  6,  4, 1}
    };

    for(int i=0;i<width*height;i++){
        dst[i]=src[i];
    }

    for(int y=2; y<height-2; y++){
        for(int x=2; x<width-2; x++){
            int sum=0;
            for(int ky=-2; ky<=2; ky++){
                for(int kx=-2; kx<=2; kx++){
                    int pixel=src[(y+ky)*width+(x+kx)];
                    int weight=kernel[ky+2][kx+2];
                    sum+=pixel*weight;
                }
            }
            dst[y*width+x]=(unsigned char)(sum/256);
        }
    }
}

/*  cost0[d,y,x]=|left[y,x] - right[y,x-d]|
    idx=d*height*width + y*width+x */
__global__ void compute_cost(const unsigned char* left, const unsigned char* right, int* cost0, 
                            int width, int height, int max_disp){
    int idx=blockIdx.x*blockDim.x + threadIdx.x;    //全局第几个线程
    int total=width*height*max_disp;
    if(idx>=total) return;   //防止多余线程越界访问
    
    int x=idx%width;
    int y=(idx/width)%height;
    int d=idx/(height*width);
    if(x>=d){
        int diff=(int)left[y*width+x]-(int)right[y*width+(x-d)];
        cost0[idx]=diff>=0 ? diff:-diff;
    }
    else{cost0[idx]=0;}  //右图越界
}

//水平方向窗口聚合
__global__ void horizontal_aggregation(const int* cost0, int* cost_h, int width, int height, int max_disp, int window_size){
    int idx=blockIdx.x*blockDim.x + threadIdx.x;
    int total=width*height*max_disp;
    if(idx>=total) return;

    int x=idx%width;
    int y=(idx/width)%height;
    int d=idx/(height*width);
    int r=window_size/2;
    if(x<r || x>=width-r){
        cost_h[idx]=0;
        return;
    }
    
    int sum=0;
    int base=d*width*height + y*width;
    for(int wx=-r; wx<=r; wx++) sum+=cost0[base+x+wx];
    cost_h[idx]=sum;
}

//垂直方向窗口聚合
__global__ void vertical_aggregation(const int* cost_h, int* cost_aggr, int width, int height, int max_disp, int window_size){
    int idx=blockIdx.x*blockDim.x + threadIdx.x;
    int total=width*height*max_disp;
    if(idx>=total) return;

    int x=idx%width;
    int y=(idx/width)%height;
    int d=idx/(height*width);
    int r=window_size/2;
    if(y<r || y>=height-r){
        cost_aggr[idx]=0;
        return;
    }

    int sum=0;
    for(int wy=-r; wy<=r; wy++){
        int cur_idx=d*width*height + (y+wy)*width + x;
        sum+=cost_h[cur_idx];
    }
    cost_aggr[idx]=sum;
}

__global__ void choose_d(const int* cost_aggr, unsigned char* disp, int width, int height, int max_disp, int window_size){
    int x=blockIdx.x*blockDim.x + threadIdx.x;
    int y=blockIdx.y*blockDim.y + threadIdx.y;
    int r=window_size/2;
    if(x>=width || y>=height) return;
    if(x<max_disp+r || x>=width-r || y<r || y>=height-r){
        disp[y*width+x]=0;
        return;
    }
    
    int best_cost=INT_MAX;
    int best_d=0;
    for(int d=0; d<max_disp; d++){
        int idx=d*width*height + y*width+x;
        int cost=cost_aggr[idx];
        if(cost<best_cost){
            best_cost=cost;
            best_d=d;
        }
    }
    disp[y*width+x]=(unsigned char)(best_d*255/max_disp);
}

int main(int argc, char* argv[]){
    //运行示例：./1serial left.pgm right.pgm disp_serial.pgm 64 9
    //disp_serial.pgm：输出视差图；64：最大视差；9：窗口大小
    if(argc<6){
        printf("Example:%s left.pgm right.pgm disp_serial.pgm 64 9\n",argv[0]);
        return 1;
    }
    const char* left_f=argv[1];
    const char* right_f=argv[2];
    const char* out_f=argv[3];
    int max_disp=atoi(argv[4]);//把字符串转为整数
    int window_size=atoi(argv[5]);

    if(max_disp<=0 || window_size<=0 || window_size%2==0){
        printf("arg error\n");
        return 1;
    }

    int width_left,height_left,width_right,height_right;
    unsigned char* left=read_pgm(left_f, &width_left, &height_left);
    unsigned char* right=read_pgm(right_f, &width_right, &height_right);
    if(left==NULL || right==NULL || width_left!=width_right || height_left!=height_right){
        printf("image error\n");
        return 1;
    }

    size_t image_bytes=(size_t)width_left*height_left*sizeof(unsigned char);
    size_t cost_bytes=(size_t)width_left*height_left*max_disp*sizeof(int);

    //unsigned char* left_smooth=(unsigned char*)malloc(width_left * height_left * sizeof(unsigned char));
    //unsigned char* right_smooth=(unsigned char*)malloc(width_left * height_left * sizeof(unsigned char));
    unsigned char* disp=(unsigned char*)malloc(width_left*height_left*sizeof(unsigned char));
    
    printf("Image size: %d x %d\n", width_left, height_left);
    printf("max_disp: %d\n", max_disp);
    printf("window_size: %d\n", window_size);

    //gaussian_blur_5x5(left, left_smooth, width_left, height_left);
    //gaussian_blur_5x5(right, right_smooth, width_left, height_left);

    unsigned char* d_left=NULL;
    unsigned char* d_right=NULL;
    unsigned char* d_disp=NULL;

    int* d_cost0=NULL;
    int* d_cost_h=NULL;
    int* d_cost_aggr=NULL;

    check_cuda(cudaMalloc((void**)&d_left, image_bytes), "cudaMalloc d_left");
    check_cuda(cudaMalloc((void**)&d_right, image_bytes), "cudaMalloc d_right");
    check_cuda(cudaMalloc((void**)&d_disp, image_bytes), "cudaMalloc d_disp");

    check_cuda(cudaMalloc((void**)&d_cost0, cost_bytes), "cudaMalloc d_cost0");
    check_cuda(cudaMalloc((void**)&d_cost_h, cost_bytes), "cudaMalloc d_cost_h");
    check_cuda(cudaMalloc((void**)&d_cost_aggr, cost_bytes), "cudaMalloc d_cost_aggr");

    // check_cuda(cudaMemcpy(d_left, left_smooth, image_size, cudaMemcpyHostToDevice), "copy left to device");
    // check_cuda(cudaMemcpy(d_right, right_smooth, image_size, cudaMemcpyHostToDevice), "copy right to device");
    check_cuda(cudaMemcpy(d_left, left, image_bytes, cudaMemcpyHostToDevice), "copy left to device");
    check_cuda(cudaMemcpy(d_right, right, image_bytes, cudaMemcpyHostToDevice), "copy right to device");
    check_cuda(cudaMemset(d_disp, 0, image_bytes), "clear d_disp");

    int total_cost_elements=width_left*height_left*max_disp;

    dim3 block1d(256);
    dim3 grid1d((total_cost_elements+block1d.x-1)/block1d.x);

    dim3 block2d(16,16);
    dim3 grid2d((width_left+block2d.x-1)/block2d.x, (height_left+block2d.y-1)/block2d.y);

    cudaEvent_t start,stop;
    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");
    check_cuda(cudaEventRecord(start), "record start");

    compute_cost<<<grid1d, block1d>>>(d_left, d_right, d_cost0, width_left, height_left, max_disp);
    check_cuda(cudaGetLastError(), "compute_cost kernel");
    horizontal_aggregation<<<grid1d, block1d>>>(d_cost0, d_cost_h, width_left, height_left, max_disp, window_size);
    check_cuda(cudaGetLastError(), "horizontal_aggregation kernel");
    vertical_aggregation<<<grid1d, block1d>>>(d_cost_h, d_cost_aggr, width_left, height_left, max_disp, window_size);
    check_cuda(cudaGetLastError(), "vertical_aggregation kernel");
    choose_d<<<grid2d, block2d>>>(d_cost_aggr, d_disp, width_left, height_left, max_disp, window_size);
    check_cuda(cudaGetLastError(), "choose_d kernel");

    check_cuda(cudaEventRecord(stop), "record stop");
    check_cuda(cudaEventSynchronize(stop), "sync stop");
    float time_ms=0.0f;
    check_cuda(cudaEventElapsedTime(&time_ms, start, stop), "time");
    check_cuda(cudaMemcpy(disp, d_disp, image_bytes, cudaMemcpyDeviceToHost), "copy disp to host");
    printf("CUDA2 StereoBM time: %.3f ms\n", time_ms);

    if(write_pgm(out_f, disp, width_left, height_left)){
        printf("Saved disparity image to %s", out_f);
    }
    else {printf("write error\n");}

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_disp);
    cudaFree(d_cost0);
    cudaFree(d_cost_h);
    cudaFree(d_cost_aggr);

    free(left);
    free(right);
    //free(left_smooth);
    //free(right_smooth);
    free(disp);

    return 0;
}

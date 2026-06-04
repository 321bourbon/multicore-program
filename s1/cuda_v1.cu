#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include "pgm_io.h"
#include <cuda_runtime.h>

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

__global__ void stereo_bm_cuda(const unsigned char* left, const unsigned char* right, unsigned char* disp,
                                int width, int height, int max_disp, int window_size){
    int x=blockIdx.x * blockDim.x + threadIdx.x;
    int y=blockIdx.y * blockDim.y + threadIdx.y;
    int r=window_size/2;
    if(x>=width || y>=height) return;
    if(x<max_disp+r || x>=width-r || y<r || y>=height-r){
        disp[y*width+x]=0;
        return;
    }
    int best_cost=INT_MAX;
    int best_d=0;
    for(int d=0; d<max_disp; d++){
        int cost=0;
        for(int wy=-r; wy<=r; wy++){
            int left_row=(y+wy)*width;
            int right_row=(y+wy)*width;
            for(int wx=-r; wx<=r; wx++){
                int left_idx=left_row+(x+wx);
                int right_idx=right_row+(x-d+wx);
                int diff=(int)left[left_idx]-(int)right[right_idx];
                cost+= diff>0 ?diff:-diff;
            }
        }
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
    
    int image_size=width_left*height_left*sizeof(unsigned char);
    unsigned char* left_smooth=(unsigned char*)malloc(width_left * height_left * sizeof(unsigned char));
    unsigned char* right_smooth=(unsigned char*)malloc(width_left * height_left * sizeof(unsigned char));
    unsigned char* disp=(unsigned char*)malloc(width_left*height_left*sizeof(unsigned char));
    
    printf("Image size: %d x %d\n", width_left, height_left);
    printf("max_disp: %d\n", max_disp);
    printf("window_size: %d\n", window_size);

    gaussian_blur_5x5(left, left_smooth, width_left, height_left);
    gaussian_blur_5x5(right, right_smooth, width_left, height_left);

    unsigned char* d_left=NULL;
    unsigned char* d_right=NULL;
    unsigned char* d_disp=NULL;

    check_cuda(cudaMalloc((void**)&d_left, image_size), "cudaMalloc d_left");
    check_cuda(cudaMalloc((void**)&d_right, image_size), "cudaMalloc d_right");
    check_cuda(cudaMalloc((void**)&d_disp, image_size), "cudaMalloc d_disp");

    // check_cuda(cudaMemcpy(d_left, left_smooth, image_size, cudaMemcpyHostToDevice), "copy left to device");
    // check_cuda(cudaMemcpy(d_right, right_smooth, image_size, cudaMemcpyHostToDevice), "copy right to device");
    check_cuda(cudaMemcpy(d_left, left, image_size, cudaMemcpyHostToDevice), "copy left to device");
    check_cuda(cudaMemcpy(d_right, right, image_size, cudaMemcpyHostToDevice), "copy right to device");
    check_cuda(cudaMemset(d_disp, 0, image_size), "clear d_disp");

    dim3 block(16,16);
    dim3 grid((width_left+block.x-1)/block.x, (height_left+block.y-1)/block.y);
    cudaEvent_t start,stop;
    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");
    check_cuda(cudaEventRecord(start), "record start");

    stereo_bm_cuda<<<grid,block>>>(d_left, d_right, d_disp, width_left, height_left, max_disp, window_size);

    check_cuda(cudaEventRecord(stop), "record stop");
    check_cuda(cudaEventSynchronize(stop), "sync stop");

    float time_ms=0.0f;
    check_cuda(cudaEventElapsedTime(&time_ms, start, stop), "time");
    check_cuda(cudaMemcpy(disp, d_disp, image_size, cudaMemcpyDeviceToHost), "copy disp to host");
    printf("CUDA StereoBM time: %.3f ms\n", time_ms);

    if(write_pgm(out_f, disp, width_left, height_left)){
        printf("Saved disparity image to %s", out_f);
    }
    else {printf("write error\n");}

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_disp);

    free(left);
    free(right);
    free(left_smooth);
    free(right_smooth);
    free(disp);

    return 0;
}
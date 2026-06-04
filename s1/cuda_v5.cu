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

__device__ unsigned int pack4_uchar(const unsigned char* img, int idx){
    unsigned int p0=(unsigned int)img[idx];
    unsigned int p1=(unsigned int)img[idx+1];
    unsigned int p2=(unsigned int)img[idx+2];
    unsigned int p3=(unsigned int)img[idx+3];

    return p0 | (p1<<8) | (p2<<16) | (p3<<24);
}

__device__ int sad4_uchar(unsigned int a, unsigned int b){
    unsigned int v=__vabsdiffu4(a,b);

    int s0=v & 0xff;
    int s1=(v>>8) & 0xff;
    int s2=(v>>16) & 0xff;
    int s3=(v>>24) & 0xff;

    return s0+s1+s2+s3;
}

__global__ void stereo_bm_cuda_sad4_early(const unsigned char* left, const unsigned char* right, unsigned char* disp,
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
        int stop=0;

        for(int wy=-r; wy<=r; wy++){
            int left_row=(y+wy)*width;
            int right_row=(y+wy)*width;

            int wx=-r;

            // 每次处理连续4个像素
            for(; wx+3<=r; wx+=4){
                int left_idx=left_row+(x+wx);
                int right_idx=right_row+(x-d+wx);

                unsigned int left_pack=pack4_uchar(left,left_idx);
                unsigned int right_pack=pack4_uchar(right,right_idx);

                cost+=sad4_uchar(left_pack,right_pack);

                if(cost>=best_cost){
                    stop=1;
                    break;
                }
            }

            if(stop){
                break;
            }

            // 剩余不足4个的像素正常计算
            for(; wx<=r; wx++){
                int left_idx=left_row+(x+wx);
                int right_idx=right_row+(x-d+wx);
                int diff=(int)left[left_idx]-(int)right[right_idx];

                cost+= diff>0 ? diff:-diff;

                if(cost>=best_cost){
                    stop=1;
                    break;
                }
            }

            if(stop){
                break;
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
    //运行示例：./cuda_v5 left.pgm right.pgm disp_cuda_v5.pgm 128 17
    //disp_cuda_v5.pgm：输出视差图；128：最大视差；17：窗口大小
    if(argc<6){
        printf("Example:%s left.pgm right.pgm disp_cuda_v5.pgm 128 17\n",argv[0]);
        return 1;
    }

    const char* left_f=argv[1];
    const char* right_f=argv[2];
    const char* out_f=argv[3];
    int max_disp=atoi(argv[4]);
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
        free(left);
        free(right);
        return 1;
    }

    int image_size=width_left*height_left*sizeof(unsigned char);
    unsigned char* disp=(unsigned char*)malloc(width_left*height_left*sizeof(unsigned char));

    if(disp==NULL){
        printf("memory error\n");
        free(left);
        free(right);
        return 1;
    }

    printf("Image size: %d x %d\n", width_left, height_left);
    printf("max_disp: %d\n", max_disp);
    printf("window_size: %d\n", window_size);

    unsigned char* d_left=NULL;
    unsigned char* d_right=NULL;
    unsigned char* d_disp=NULL;

    check_cuda(cudaMalloc((void**)&d_left, image_size), "cudaMalloc d_left");
    check_cuda(cudaMalloc((void**)&d_right, image_size), "cudaMalloc d_right");
    check_cuda(cudaMalloc((void**)&d_disp, image_size), "cudaMalloc d_disp");

    check_cuda(cudaMemcpy(d_left, left, image_size, cudaMemcpyHostToDevice), "copy left to device");
    check_cuda(cudaMemcpy(d_right, right, image_size, cudaMemcpyHostToDevice), "copy right to device");
    check_cuda(cudaMemset(d_disp, 0, image_size), "clear d_disp");

    dim3 block(16,16);
    dim3 grid((width_left+block.x-1)/block.x, (height_left+block.y-1)/block.y);

    cudaEvent_t start,stop;
    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");
    check_cuda(cudaEventRecord(start), "record start");

    stereo_bm_cuda_sad4_early<<<grid,block>>>(d_left, d_right, d_disp,
                                        width_left, height_left, max_disp, window_size);
    check_cuda(cudaGetLastError(), "kernel launch");

    check_cuda(cudaEventRecord(stop), "record stop");
    check_cuda(cudaEventSynchronize(stop), "sync stop");

    float time_ms=0.0f;
    check_cuda(cudaEventElapsedTime(&time_ms, start, stop), "time");
    check_cuda(cudaMemcpy(disp, d_disp, image_size, cudaMemcpyDeviceToHost), "copy disp to host");

    printf("CUDA5 SAD4 Early StereoBM time: %.3f ms\n", time_ms);

    if(write_pgm(out_f, disp, width_left, height_left)){
        printf("Saved disparity image to %s\n", out_f);
    }
    else{
        printf("write error\n");
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_disp);

    free(left);
    free(right);
    free(disp);

    return 0;
}
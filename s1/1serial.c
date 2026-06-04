#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include "pgm_io.h"

static int abs_int(int x){
    return x<0 ? -x:x;
}

//pgm是8位灰度图，像素灰度值范围为0-255，正好对应unsigned char，用int存的话太浪费
void stereo_bm_serial(const unsigned char* left, const unsigned char* right, unsigned char* disp,
                    int width, int height, int max_disp, int window_size){
    int r=window_size/2;
    //初始化视差图（全黑）
    for(int i=0; i<width*height; i++){
        disp[i]=0;
    }

    for(int y=r; y<height-r; y++){
        for(int x=max_disp+r; x<width-r; x++){
            int best_cost=INT_MAX;
            int best_d=0;
            //遍历视差d
            for(int d=0; d<max_disp; d++){
                //cost算的是一个窗口内的像素差总和
                int cost=0;
                //wy为窗口纵向偏移
                for(int wy=-r; wy<=r; wy++){
                    int left_row=(y+wy)*width;
                    int right_row=(y+wy)*width;
                    //wx为窗口横向偏移
                    for(int wx=-r; wx<=r; wx++){
                        //图像是用一维数组存的，所以要把二维坐标转为一维
                        int left_idx=left_row+(x+wx);
                        int right_idx=right_row+(x-d+wx);
                        int diff=(int)left[left_idx]-(int)right[right_idx];
                        cost+=abs_int(diff);
                    }
                }
                if(cost<best_cost){
                    best_cost=cost;
                    best_d=d;
                }
            }
            //把视差线性放大到0-255
            disp[x+y*width]=(unsigned char)(best_d*255/max_disp);
        }
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
                    int pixel=src[(x+kx)+(y+ky)*width];
                    int weight=kernel[ky+2][kx+2];
                    sum+=pixel*weight;
                }
            }
            dst[x+y*width]=(unsigned char)(sum/256);
        }
    }
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
    
    unsigned char* left_smooth=(unsigned char*)malloc(width_left * height_left * sizeof(unsigned char));
    unsigned char* right_smooth=(unsigned char*)malloc(width_left * height_left * sizeof(unsigned char));
    unsigned char* disp=(unsigned char*)malloc(width_left*height_left*sizeof(unsigned char));
    
    printf("Image size: %d x %d\n", width_left, height_left);
    printf("max_disp: %d\n", max_disp);
    printf("window_size: %d\n", window_size);

    gaussian_blur_5x5(left, left_smooth, width_left, height_left);
    gaussian_blur_5x5(right, right_smooth, width_left, height_left);

    clock_t start=clock();
    stereo_bm_serial(left, right, disp, width_left, height_left, max_disp, window_size);
    //stereo_bm_serial(left_smooth, right_smooth, disp, width_left, height_left, max_disp, window_size);
    clock_t end=clock();
    
    double time_ms=(double)(end-start)*1000.0/CLOCKS_PER_SEC;
    printf("Serial StereoBM time: %.3f ms\n",time_ms);

    if(write_pgm(out_f, disp, width_left, height_left)){
        printf("Saved disparity image to %s", out_f);
    }
    else {printf("write error\n");}

    free(left);
    free(right);
    free(left_smooth);
    free(right_smooth);
    free(disp);
    return 0;
}
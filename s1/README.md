# StereoBM CUDA并行实验说明

本项目实现双目立体视觉中的StereoBM块匹配算法，并对其进行CUDA并行优化。输入为左右灰度图，输出为视差图。

## 1.算法基本思路

对左图中的每个有效像素点，在右图同一行搜索不同水平偏移量。对于每个候选视差，比较左图局部窗口和右图对应偏移窗口内像素的灰度差异，并将差异累加为匹配代价。最后选择匹配代价最小的视差作为当前像素的视差值。

## 2.文件说明

```text
pgm_io.c / pgm_io.h      PGM灰度图读写
1serial.c                CPU串行版本
cuda_v1.cu               基础CUDA并行版本
cuda_v2.cu               代价体加水平、垂直聚合版本
cuda_v3.cu               Shared Memory左图缓存版本
cuda_v4.cu               四像素矢量化版本
cuda_v5.cu               矢量化加早停剪枝版本
run_all_versions.sh      批量运行脚本
conesH/im2.ppm           数据集原始左图
conesH/im6.ppm           数据集原始右图
conesH/left.pgm          im2.ppm转换得到的灰度左图
conesH/right.pgm         im6.ppm转换得到的灰度右图
conesH/disp2.pgm         数据集自带的左图参考真实视差图
conesH/disp6.pgm         数据集自带的右图参考真实视差图

```

## 3.运行环境

| 项目           | 配置                          |
| ------------ | --------------------------- |
| 操作系统         | Ubuntu 22.04.5 LTS          |
| 内核版本         | Linux 6.8.0                 |
| CPU          | Intel Xeon Platinum 8175M   |
| CPU核心数       | 48物理核心，96逻辑CPU              |
| 内存           | 125GB                       |
| GPU          | NVIDIA GeForce RTX 4090 × 2 |
| 显存           | 每张约24GB                     |
| NVIDIA驱动     | 580.95.05                   |
| CUDA Toolkit | 12.6                        |
| nvcc版本       | 12.6.20                     |
| GCC版本        | 11.4.0                      |

## 4.版本说明

### Serial

CPU单线程版本，逐个像素顺序计算视差，用作正确性和加速比基准。

```bash
gcc 1serial.c pgm_io.c -O2 -o serial
./serial conesH/left.pgm conesH/right.pgm conesH/disp_serial.pgm 128 17
```

### CUDA V1

基础CUDA版本。一个CUDA线程负责一个像素点，线程内部遍历所有候选视差和窗口像素。该版本主要将串行代码中的像素循环并行化。

```bash
nvcc cuda_v1.cu pgm_io.c -O2 -o cuda_v1
./cuda_v1 conesH/left.pgm conesH/right.pgm conesH/disp_v1.pgm 128 17
```

### CUDA V2

代价体聚合版本。先计算每个像素在每个视差下的单点代价，再分别进行水平窗口聚合和垂直窗口聚合，最后选择最优视差。该版本减少了重复窗口计算，是当前最快版本。

```bash
nvcc cuda_v2.cu pgm_io.c -O2 -o cuda_v2
./cuda_v2 conesH/left.pgm conesH/right.pgm conesH/disp_v2.pgm 128 17
```

### CUDA V3

Shared Memory左图缓存版本。每个block先把左图局部区域加载到shared memory，再进行窗口匹配。实验中该版本没有明显加速，主要原因是右图仍需大量global memory访问，同时shared memory加载和同步也有额外开销。

```bash
nvcc cuda_v3.cu pgm_io.c -O2 -o cuda_v3
./cuda_v3 conesH/left.pgm conesH/right.pgm conesH/disp_v3.pgm 128 17
```

### CUDA V4

四像素矢量化版本。在V1基础上，将横向连续4个灰度像素打包处理，减少窗口内部逐像素差异计算的指令数量。

```bash
nvcc cuda_v4.cu pgm_io.c -O2 -arch=sm_89 -o cuda_v4
./cuda_v4 conesH/left.pgm conesH/right.pgm conesH/disp_v4.pgm 128 17
```

### CUDA V5

V4加早停剪枝版本。当某个候选视差的代价在累加过程中已经不小于当前最优代价时，提前结束该候选视差的后续计算。该版本快于V1和V4，但慢于V2。

```bash
nvcc cuda_v5.cu pgm_io.c -O2 -arch=sm_89 -o cuda_v5
./cuda_v5 conesH/left.pgm conesH/right.pgm conesH/disp_v5.pgm 128 17
```

## 5.批量运行

运行脚本会自动编译Serial和CUDA V1到V5，每个版本重复运行5次，并保存日志和结果图。

```bash
chmod +x run_all_versions.sh
./run_all_versions.sh
```

结果会保存在：

```text
results/run_时间戳/
```

其中：

```text
log.txt        完整运行日志
summary.csv    每次运行时间
serial/        串行版本输出视差图
v1/到v5/       各CUDA版本输出视差图
```

## 6.当前实验结果
结果见 s1/results/run_20260529_160321

实验参数：

```text
图像大小：900 × 750
max_disp：128
window_size：17
重复次数：5
```

平均运行时间：

| 版本      |         平均时间 |
| ------- | -----------: |
| Serial  | 14547.707 ms |
| CUDA V1 |     9.366 ms |
| CUDA V2 |     3.080 ms |
| CUDA V3 |     9.417 ms |
| CUDA V4 |     8.399 ms |
| CUDA V5 |     6.295 ms |

输出一致性检查结果：

```text
serial and v1 outputs are exactly the same.
serial and v2 outputs are exactly the same.
serial and v3 outputs are exactly the same.
serial and v4 outputs are exactly the same.
serial and v5 outputs are exactly the same.
```

## 7.实验结论

- Serial版本作为CPU单线程基线，运行时间最长
- CUDA V1通过像素级线程映射实现基础并行，已经获得明显加速。
- CUDA V2通过构建代价体，并采用水平、垂直方向聚合减少重复窗口计算，取得当前最佳性能。
- CUDA V4和CUDA V5从线程内部计算入手，通过四像素打包和早停剪枝进一步降低内层计算开销。
- CUDA V3说明Shared Memory并不一定带来性能提升，实际效果取决于访存模式、同步开销和硬件缓存机制。

本实验最终采用CUDA V2作为性能最优版本，CUDA V5作为另一条有效优化路线。





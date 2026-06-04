#!/bin/bash

RUN_DIR="results/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

LOG_FILE="$RUN_DIR/log.txt"

{
echo "Result directory: $RUN_DIR"
echo ""

echo "===== Compile CUDA V1 and V4 ====="
nvcc 2cuda.cu pgm_io.c -O2 -o cuda_v1
nvcc 5cuda.cu pgm_io.c -O2 -arch=sm_89 -o cuda_v4

echo ""
echo "===== CUDA V1: max_disp=128, window_size=17 ====="
for i in 1 2 3 4 5
do
    echo "Run $i"
    ./cuda_v1 conesH/left.pgm conesH/right.pgm "$RUN_DIR/disp_cuda_v1_run${i}.pgm" 128 17
done

echo ""
echo "===== CUDA V4: max_disp=128, window_size=17 ====="
for i in 1 2 3 4 5
do
    echo "Run $i"
    ./cuda_v4 conesH/left.pgm conesH/right.pgm "$RUN_DIR/disp_cuda_v4_run${i}.pgm" 128 17
done

echo ""
echo "===== Compare output of last run ====="
cmp "$RUN_DIR/disp_cuda_v1_run5.pgm" "$RUN_DIR/disp_cuda_v4_run5.pgm"

if [ $? -eq 0 ]; then
    echo "V1 and V4 outputs are exactly the same."
else
    echo "V1 and V4 outputs are different."
fi

echo ""
echo "All results saved in: $RUN_DIR"
} | tee "$LOG_FILE"
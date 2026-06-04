#!/bin/bash
# 敏感性实验：window_size参数测试（V1, V2, V5）
# 固定：max_disp=128，图像=conesH

set -e

# 参数配置
LEFT_IMG="conesH/left.pgm"
RIGHT_IMG="conesH/right.pgm"
MAX_DISP=128
WINDOW_SIZE_LIST=(5 9 13 17 21)
RUNS=5

# 输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="results/sensitivity_window_size_${TIMESTAMP}"
mkdir -p "$RESULT_DIR"

# 编译（如果还未编译）
echo "===== Compiling CUDA versions ====="
[ ! -f cuda_v1 ] && nvcc cuda_v1.cu pgm_io.c -O2 -o cuda_v1
[ ! -f cuda_v2 ] && nvcc cuda_v2.cu pgm_io.c -O2 -o cuda_v2
[ ! -f cuda_v5 ] && nvcc cuda_v5.cu pgm_io.c -O2 -arch=sm_89 -o cuda_v5

# 运行实验
CSV_FILE="$RESULT_DIR/summary.csv"
echo "window_size,version,run,time_ms" > "$CSV_FILE"

for ws in "${WINDOW_SIZE_LIST[@]}"; do
    echo ""
    echo "===== Testing window_size=$ws ====="
    
    for version in "v1" "v2" "v5"; do
        echo "--- Running $version ---"
        
        exec_name="cuda_${version}"
        [ ! -f "$exec_name" ] && echo "Warning: $exec_name not found" && continue
        
        out_img="$RESULT_DIR/disp_${version}_ws${ws}_temp.pgm"
        
        for run in $(seq 1 $RUNS); do
            echo -n "  Run $run/$RUNS: "
            
            output=$(./"$exec_name" "$LEFT_IMG" "$RIGHT_IMG" "$out_img" "$MAX_DISP" "$ws" 2>&1)
            time_str=$(printf '%s\n' "$output" | awk -F'time: ' '/time: /{print $2; exit}' | awk '{print $1}')
            [ -z "$time_str" ] && time_str="N/A"
            
            echo "$time_str ms"
            echo "$ws,$version,$run,$time_str" >> "$CSV_FILE"
        done
        
        rm -f "$out_img"
    done
done

echo ""
echo "===== Experiment Complete ====="
echo "Results saved to: $RESULT_DIR"

# 分析
cat > "${RESULT_DIR}/analysis.txt" << 'EOF'
Window_Size Sensitivity Analysis:

For each (window_size, version) pair:
- Average time across 5 runs
- Speedup = time(v1) / time(version)
- Plot: window_size vs time (should follow quadratic trend)

Expected Pattern:
- Time ≈ 0.5 * max_disp * (2*r+1)^2 operations per pixel
- window_size=5: (2*2+1)^2 = 25
- window_size=21: (2*10+1)^2 = 441
- So time should increase by ~17.6x (441/25)
EOF

echo "Next: Import $RESULT_DIR/summary.csv to Excel/Python for analysis"

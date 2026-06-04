#!/bin/bash
# 敏感性实验：max_disp参数测试（V1, V2, V5）
# 固定：window_size=17，图像=conesH

set -e

# 参数配置
LEFT_IMG="conesH/left.pgm"
RIGHT_IMG="conesH/right.pgm"
WINDOW_SIZE=17
MAX_DISP_LIST=(32 64 96 128)
RUNS=5

# 输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="results/sensitivity_max_disp_${TIMESTAMP}"
mkdir -p "$RESULT_DIR"

# 编译（如果还未编译）
echo "===== Compiling CUDA versions ====="
if [ ! -f cuda_v1 ]; then
    nvcc cuda_v1.cu pgm_io.c -O2 -o cuda_v1 || echo "V1 compile failed"
fi
if [ ! -f cuda_v2 ]; then
    nvcc cuda_v2.cu pgm_io.c -O2 -o cuda_v2 || echo "V2 compile failed"
fi
if [ ! -f cuda_v5 ]; then
    nvcc cuda_v5.cu pgm_io.c -O2 -arch=sm_89 -o cuda_v5 || echo "V5 compile failed"
fi

# 运行实验
LOG_FILE="$RESULT_DIR/log.txt"
CSV_FILE="$RESULT_DIR/summary.csv"

echo "max_disp,version,run,time_ms" > "$CSV_FILE"

for max_disp in "${MAX_DISP_LIST[@]}"; do
    echo ""
    echo "===== Testing max_disp=$max_disp ====="
    
    for version in "v1" "v2" "v5"; do
        echo "--- Running $version ---"
        
        # 检查可执行文件
        exec_name="cuda_${version}"
        if [ ! -f "$exec_name" ]; then
            echo "Warning: $exec_name not found, skipping"
            continue
        fi
        
        out_img="$RESULT_DIR/disp_${version}_max${max_disp}_temp.pgm"
        
        for run in $(seq 1 $RUNS); do
            echo -n "  Run $run/$RUNS: "
            
            # 运行并提取时间（假设输出格式为 "CUDA StereoBM time: XXX.XXX ms"）
            output=$(./"$exec_name" "$LEFT_IMG" "$RIGHT_IMG" "$out_img" "$max_disp" "$WINDOW_SIZE" 2>&1)
            time_str=$(printf '%s\n' "$output" | awk -F'time: ' '/time: /{print $2; exit}' | awk '{print $1}')
            [ -z "$time_str" ] && time_str="N/A"
            
            echo "$time_str ms"
            echo "$max_disp,$version,$run,$time_str" >> "$CSV_FILE"
        done
        
        # 清理临时文件
        rm -f "$out_img"
    done
done

echo ""
echo "===== Experiment Complete ====="
echo "Results saved to: $RESULT_DIR"
echo ""

# 生成统计摘要
echo "===== Summary Statistics ====="
cat > "${RESULT_DIR}/analysis.txt" << 'EOF'
Max_Disp Sensitivity Analysis Summary:

For each (max_disp, version) pair:
- Compute average time across 5 runs
- Compute speedup = time(v1) / time(version)
- Plot: max_disp vs time (line chart with v1/v2/v5)

Expected Pattern:
- Time should grow roughly as O(max_disp * window_size^2)
- V2 should be ~3x faster than V1
- V5 should be ~1.5x faster than V1
EOF

echo "Summary saved. Check $RESULT_DIR/analysis.txt for next steps."

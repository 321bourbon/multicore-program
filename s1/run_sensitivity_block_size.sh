#!/bin/bash
# 敏感性实验：CUDA block大小测试（仅V5）
# 固定：max_disp=128, window_size=17, 图像=conesH

set -e

# 参数配置
LEFT_IMG="conesH/left.pgm"
RIGHT_IMG="conesH/right.pgm"
MAX_DISP=128
WINDOW_SIZE=17
BLOCK_CONFIGS=("8,8" "16,16" "16,32" "32,16")
RUNS=5

# 输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="results/sensitivity_block_size_${TIMESTAMP}"
mkdir -p "$RESULT_DIR"

# 复制原始cuda_v5.cu为模板（备份）
cp cuda_v5.cu cuda_v5_template.cu

CSV_FILE="$RESULT_DIR/summary.csv"
echo "block_size,run,time_ms" > "$CSV_FILE"

for block_config in "${BLOCK_CONFIGS[@]}"; do
    IFS=',' read -r BX BY <<< "$block_config"
    BLOCK_STR="${BX}x${BY}"
    
    echo ""
    echo "===== Testing block=$BLOCK_STR ====="
    
    # 根据block配置生成cuda_v5的变体
    sed "s/dim3 block(16,16)/dim3 block($BX,$BY)/" cuda_v5_template.cu > cuda_v5_var.cu
    
    # 编译
    nvcc_cmd="nvcc cuda_v5_var.cu pgm_io.c -O2 -arch=sm_89 -o cuda_v5_test"
    echo "Compiling: $nvcc_cmd"
    
    if ! $nvcc_cmd 2>&1; then
        echo "Compile failed for block=$BLOCK_STR, skipping"
        continue
    fi
    
    out_img="$RESULT_DIR/disp_block${BLOCK_STR}_temp.pgm"
    
    for run in $(seq 1 $RUNS); do
        echo -n "  Run $run/$RUNS (block=$BLOCK_STR): "
        
        output=$(./cuda_v5_test "$LEFT_IMG" "$RIGHT_IMG" "$out_img" "$MAX_DISP" "$WINDOW_SIZE" 2>&1)
        time_str=$(printf '%s\n' "$output" | awk -F'time: ' '/time: /{print $2; exit}' | awk '{print $1}')
        [ -z "$time_str" ] && time_str="N/A"
        
        echo "$time_str ms"
        echo "$BLOCK_STR,$run,$time_str" >> "$CSV_FILE"
    done
    
    rm -f "$out_img"
done

# 清理
rm -f cuda_v5_template.cu cuda_v5_var.cu cuda_v5_test

echo ""
echo "===== Experiment Complete ====="
echo "Results saved to: $RESULT_DIR/summary.csv"
echo ""

# 分析说明
cat > "${RESULT_DIR}/analysis.txt" << 'EOF'
Block Size Sensitivity Analysis (CUDA V5 only):

Tested configurations:
- 8×8: 64 threads per block
- 16×16: 256 threads per block  
- 16×32: 512 threads per block
- 32×16: 512 threads per block

Expected observations:
1. More threads per block → better occupancy (generally faster)
2. But memory access patterns may affect actual performance
3. RTX 4090 has high occupancy, so difference might be small (<5%)

Report best block size in paper section on "Implementation Details"
EOF

echo "For analysis: python3 -c \"import pandas as pd; df=pd.read_csv('$CSV_FILE'); print(df.groupby('block_size')['time_ms'].mean())\""

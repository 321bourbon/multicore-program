#!/bin/bash
# 多数据集综合测试：V1, V2, V5在不同图像尺寸下的性能
# 需要先下载数据集（见下方说明）

set -e

# ======== 数据集配置 ========
# 下载步骤（在Linux服务器上执行一次）：
# wget -r https://vision.middlebury.edu/stereo/data/2001/MiddleburyData/
# wget -r https://vision.middlebury.edu/stereo/data/2003/MiddleburyData/
# 然后将pgm文件按如下结构放置

declare -A DATASETS
DATASETS=(
    ["Tsukuba"]="tsukuba_stereo/frame00/im0.png|tsukuba_stereo/frame00/im1.png|384|288"
    ["Teddy"]="teddy_stereo/frame10/im2.png|teddy_stereo/frame10/im6.png|640|480"
    ["ConesH"]="conesH/left.pgm|conesH/right.pgm|900|750"
    ["Art"]="art_stereo/frame00/im0.png|art_stereo/frame00/im1.png|1024|768"
)

# 若图像为PNG格式，需要先转为灰度PGM（可选脚本见下方）

# ======== 实验配置 ========
MAX_DISP=128
WINDOW_SIZE=17
RUNS=5

# 输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="results/multi_dataset_${TIMESTAMP}"
mkdir -p "$RESULT_DIR"

# ======== 编译 ========
echo "===== Compiling CUDA versions ====="
[ ! -f cuda_v1 ] && nvcc cuda_v1.cu pgm_io.c -O2 -o cuda_v1
[ ! -f cuda_v2 ] && nvcc cuda_v2.cu pgm_io.c -O2 -o cuda_v2
[ ! -f cuda_v5 ] && nvcc cuda_v5.cu pgm_io.c -O2 -arch=sm_89 -o cuda_v5

# ======== 运行实验 ========
CSV_FILE="$RESULT_DIR/summary.csv"
echo "dataset,image_size,version,run,time_ms" > "$CSV_FILE"

for dataset_name in "${!DATASETS[@]}"; do
    IFS='|' read -r left_path right_path width height <<< "${DATASETS[$dataset_name]}"
    img_size="${width}x${height}"
    
    # 检查文件存在
    if [ ! -f "$left_path" ] || [ ! -f "$right_path" ]; then
        echo ""
        echo "⚠️  Dataset $dataset_name not found at $left_path / $right_path"
        echo "   Skipping..."
        continue
    fi
    
    echo ""
    echo "===== Testing $dataset_name ($img_size) ====="
    
    for version in "v1" "v2" "v5"; do
        echo "--- Running $version ---"
        
        exec_name="cuda_${version}"
        [ ! -f "$exec_name" ] && continue
        
        out_dir="$RESULT_DIR/$dataset_name/$version"
        mkdir -p "$out_dir"
        out_img="$out_dir/disp_temp.pgm"
        
        for run in $(seq 1 $RUNS); do
            echo -n "  Run $run/$RUNS: "
            
            output=$(./"$exec_name" "$left_path" "$right_path" "$out_img" "$MAX_DISP" "$WINDOW_SIZE" 2>&1)
            time_str=$(printf '%s\n' "$output" | awk -F'time: ' '/time: /{print $2; exit}' | awk '{print $1}')
            [ -z "$time_str" ] && time_str="N/A"
            
            echo "$time_str ms"
            echo "$dataset_name,$img_size,$version,$run,$time_str" >> "$CSV_FILE"
        done
    done
done

echo ""
echo "===== Experiment Complete ====="
echo "Results: $CSV_FILE"

# ======== 生成分析报告 ========
cat > "${RESULT_DIR}/README.txt" << 'EOF'
Multi-Dataset Sensitivity Analysis Results

CSV Format: dataset, image_size, version, run, time_ms

Analysis Steps:
1. Group by (dataset, version) and compute average time
2. Compute speedup for each version relative to V1
3. Create line chart: dataset_size vs avg_time with V1/V2/V5 three lines
4. Create bar chart: speedup comparison across datasets

Expected Findings:
- V2 should consistently ~3x faster than V1 across all datasets
- V5 should be ~1.5x faster than V1
- Larger images → higher absolute time but similar speedup ratio
EOF

echo "Next step: python3 analysis_script.py $CSV_FILE (see template below)"

# Python分析模板
cat > "${RESULT_DIR}/analysis_template.py" << 'EOF'
import pandas as pd
import sys

if len(sys.argv) > 1:
    csv_file = sys.argv[1]
else:
    csv_file = "summary.csv"

df = pd.read_csv(csv_file)

# 分组计算平均时间和加速比
summary = df.groupby(['dataset', 'image_size', 'version'])['time_ms'].agg(['mean', 'std']).reset_index()
summary.columns = ['dataset', 'image_size', 'version', 'avg_time_ms', 'std_dev']

# 计算加速比
v1_times = summary[summary['version'] == 'v1'][['dataset', 'avg_time_ms']].rename(columns={'avg_time_ms': 'v1_time'})
summary = summary.merge(v1_times, on='dataset')
summary['speedup'] = summary['v1_time'] / summary['avg_time_ms']

print(summary.to_string(index=False))
print("\n" + "="*60)
print("Summary Statistics:")
print(summary.groupby('version')[['avg_time_ms', 'speedup']].mean())
EOF

chmod +x "${RESULT_DIR}/analysis_template.py"
echo ""
echo "Generated analysis script at: ${RESULT_DIR}/analysis_template.py"

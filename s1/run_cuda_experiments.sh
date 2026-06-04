#!/bin/bash
# run_cuda_experiments.sh
#
# 这个脚本用于自动编译并运行 CUDA StereoBM 实验，
# 包括 max_disp 敏感性、window_size 敏感性和 V5 block 大小敏感性，
# 每个参数组合运行多次并输出平均时间。

set -euo pipefail

# 实验参数：可以按需修改
RUNS=5
LEFT_IMG="conesH/left.pgm"
RIGHT_IMG="conesH/right.pgm"
WINDOW_SIZE_LIST=(5 9 13 17 21)
MAX_DISP_LIST=(32 64 96 128)
BLOCK_CONFIGS=("8 8" "16 16" "16 32" "32 16")
VERSIONS=("v1" "v2" "v5")

# 结果存放目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="results/cuda_experiments_${TIMESTAMP}"
mkdir -p "$RESULT_DIR"

# 记录运行环境信息
cat > "$RESULT_DIR/environment.txt" <<EOF
Run timestamp: $TIMESTAMP
RUNS: $RUNS
LEFT_IMG: $LEFT_IMG
RIGHT_IMG: $RIGHT_IMG
WINDOW_SIZE_LIST: ${WINDOW_SIZE_LIST[*]}
MAX_DISP_LIST: ${MAX_DISP_LIST[*]}
BLOCK_CONFIGS: ${BLOCK_CONFIGS[*]}
VERSIONS: ${VERSIONS[*]}
EOF

# 1. 编译函数：如果可执行文件不存在则编译
compile_version() {
    local version="$1"
    local src_file="cuda_${version}.cu"
    local exe_file="cuda_${version}"

    if [ -f "$exe_file" ]; then
        echo "[compile] $exe_file already exists, skipping compile."
        return
    fi

    echo "[compile] Building $exe_file from $src_file..."
    if [ "$version" = "v5" ]; then
        nvcc "$src_file" pgm_io.c -O2 -arch=sm_89 -o "$exe_file"
    else
        nvcc "$src_file" pgm_io.c -O2 -o "$exe_file"
    fi
}

# 2. 运行命令并提取时间，返回 time_ms
run_and_extract_time() {
    local cmd="$1"
    local output

    output=$(eval "$cmd" 2>&1)
    printf '%s\n' "$output" | sed -n 's/.*time: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n 1
}

# 3. 实验执行函数：针对普通参数实验（max_disp 或 window_size）
run_experiment_set() {
    local label="$1"
    local param_name="$2"
    local param_values=(${!3})
    local fixed_arg_name="$4"
    local fixed_arg_value="$5"
    local csv_file="$RESULT_DIR/${label}.csv"

    echo "[experiment] Running ${label} sensitivity test"
    echo "${param_name},version,run,time_ms" > "$csv_file"

    for param_value in "${param_values[@]}"; do
        echo "--- ${param_name} = ${param_value} ---"

        for version in "${VERSIONS[@]}"; do
            local exe_file="cuda_${version}"
            if [ ! -x "$exe_file" ]; then
                echo "[warning] Executable $exe_file not found, skipping"
                continue
            fi

            for run_idx in $(seq 1 "$RUNS"); do
                local out_img="$RESULT_DIR/${label}_${version}_${param_value}_run${run_idx}.pgm"
                local cmd="./$exe_file $LEFT_IMG $RIGHT_IMG $out_img"
                if [ "$param_name" = "max_disp" ]; then
                    cmd+=" $param_value $fixed_arg_value"
                else
                    cmd+=" $fixed_arg_value $param_value"
                fi

                echo -n "[run] $version ${param_name}=${param_value} run=${run_idx} ... "
                local time_ms
                time_ms=$(run_and_extract_time "$cmd")
                if [ -z "$time_ms" ]; then
                    echo "FAILED"
                    echo "[error] no time output from command: $cmd"
                    exit 1
                fi
                echo "$time_ms ms"
                echo "$param_value,$version,$run_idx,$time_ms" >> "$csv_file"
                rm -f "$out_img"
            done
        done
    done
}

# 4. Block 大小实验，专用于 V5
run_block_experiment() {
    local label="block_size"
    local csv_file="$RESULT_DIR/${label}.csv"

    echo "[experiment] Running block size test for V5"
    echo "block_size,run,time_ms" > "$csv_file"

    local exe_file="cuda_v5"
    if [ ! -x "$exe_file" ]; then
        echo "[warning] Executable $exe_file not found, skipping block experiment"
        return
    fi

    for config in "${BLOCK_CONFIGS[@]}"; do
        IFS=' ' read -r bx by <<< "$config"
        local block_label="${bx}x${by}"
        local temp_src="${RESULT_DIR}/cuda_v5_block_${block_label}.cu"
        local temp_exe="${RESULT_DIR}/cuda_v5_block_${block_label}"

        echo "--- block=$block_label ---"

        # 复制原始源文件并替换 block 配置
        sed "s/dim3 block(16,16)/dim3 block(${bx},${by})/" cuda_v5.cu > "$temp_src"
        nvcc "$temp_src" pgm_io.c -O2 -arch=sm_89 -o "$temp_exe"

        for run_idx in $(seq 1 "$RUNS"); do
            local out_img="$RESULT_DIR/block_${block_label}_run${run_idx}.pgm"
            local cmd="./${temp_exe} $LEFT_IMG $RIGHT_IMG $out_img 128 17"

            echo -n "[run] V5 block=${block_label} run=${run_idx} ... "
            local time_ms
            time_ms=$(run_and_extract_time "$cmd")
            if [ -z "$time_ms" ]; then
                echo "FAILED"
                echo "[error] no time output from command: $cmd"
                exit 1
            fi
            echo "$time_ms ms"
            echo "$block_label,$run_idx,$time_ms" >> "$csv_file"
            rm -f "$out_img"
        done

        rm -f "$temp_src" "$temp_exe"
    done
}

# 5. 计算平均值的辅助函数
write_average_results() {
    local csv_file="$1"
    local summary_file="${csv_file%.csv}_average.txt"

    awk -F',' 'NR>1 {count[$1","$2]++; sum[$1","$2]+=$4} END {for (k in sum) printf "%s,%.4f\n", k, sum[k]/count[k]}' "$csv_file" | sort > "$summary_file"
    echo "[summary] Average results written to $summary_file"
}

# 6. 执行所有实验
for version in "${VERSIONS[@]}"; do
    compile_version "$version"
done

# max_disp 敏感性：固定 window_size=17
run_experiment_set "max_disp" "max_disp" MAX_DISP_LIST[@] "window_size" 17
# window_size 敏感性：固定 max_disp=128
run_experiment_set "window_size" "window_size" WINDOW_SIZE_LIST[@] "max_disp" 128
# block 大小敏感性：仅 CUDA V5
run_block_experiment

# 7. 生成平均值结果
write_average_results "$RESULT_DIR/max_disp.csv"
write_average_results "$RESULT_DIR/window_size.csv"
write_average_results "$RESULT_DIR/block_size.csv"

echo ""
echo "All experiments finished. Results are saved in: $RESULT_DIR"

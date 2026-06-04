#!/bin/bash
set -e

N=5
MAX_DISP=128
WINDOW_SIZE=17

LEFT_IMG="conesH/left.pgm"
RIGHT_IMG="conesH/right.pgm"

RUN_DIR="results/run_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$RUN_DIR/log.txt"
SUMMARY_FILE="$RUN_DIR/summary.csv"

mkdir -p "$RUN_DIR"

{
echo "Result directory: $RUN_DIR"
echo "max_disp=$MAX_DISP"
echo "window_size=$WINDOW_SIZE"
echo "repeat=$N"
echo ""

echo "===== Compile all versions ====="
gcc 1serial.c pgm_io.c -O2 -o serial
nvcc cuda_v1.cu pgm_io.c -O2 -o cuda_v1
nvcc cuda_v2.cu pgm_io.c -O2 -o cuda_v2
nvcc cuda_v3.cu pgm_io.c -O2 -o cuda_v3
nvcc cuda_v4.cu pgm_io.c -O2 -arch=sm_89 -o cuda_v4
nvcc cuda_v5.cu pgm_io.c -O2 -arch=sm_89 -o cuda_v5

echo ""
echo "version,run,time_ms" > "$SUMMARY_FILE"

run_version(){
    VERSION_NAME=$1
    EXEC_FILE=$2

    VERSION_DIR="$RUN_DIR/$VERSION_NAME"
    mkdir -p "$VERSION_DIR"

    echo ""
    echo "===== $VERSION_NAME ====="

    for i in $(seq 1 $N)
    do
        OUT_FILE="$VERSION_DIR/disp_${VERSION_NAME}_run${i}.pgm"

        echo "Run $i"
        OUTPUT=$($EXEC_FILE "$LEFT_IMG" "$RIGHT_IMG" "$OUT_FILE" $MAX_DISP $WINDOW_SIZE)
        echo "$OUTPUT"

        TIME_MS=$(echo "$OUTPUT" | grep "time:" | awk -F'time: ' '{print $2}' | awk '{print $1}')
        echo "$VERSION_NAME,$i,$TIME_MS" >> "$SUMMARY_FILE"
    done
}

run_version "serial" "./serial"
run_version "v1" "./cuda_v1"
run_version "v2" "./cuda_v2"
run_version "v3" "./cuda_v3"
run_version "v4" "./cuda_v4"
run_version "v5" "./cuda_v5"

echo ""
echo "===== Average time ====="
awk -F',' '
NR>1{
    sum[$1]+=$3;
    cnt[$1]++;
}
END{
    for(v in sum){
        printf "%s average: %.3f ms\n", v, sum[v]/cnt[v];
    }
}
' "$SUMMARY_FILE" | sort

echo ""
echo "===== Output consistency check ====="

BASE="$RUN_DIR/serial/disp_serial_run${N}.pgm"

for v in v1 v2 v3 v4 v5
do
    TARGET="$RUN_DIR/$v/disp_${v}_run${N}.pgm"
    if cmp -s "$BASE" "$TARGET"; then
        echo "serial and $v outputs are exactly the same."
    else
        echo "serial and $v outputs are different."
    fi
done

echo ""
echo "All results saved in: $RUN_DIR"
echo "Summary saved in: $SUMMARY_FILE"

} | tee "$LOG_FILE"
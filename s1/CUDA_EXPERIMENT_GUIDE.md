# CUDA 敏感性实验执行指南

## 📋 概览

你需要完成以下 4 组实验，通过这些脚本自动化完成：

| 实验 | 脚本 | 固定参数 | 变化参数 | 说明 |
|------|------|--------|--------|------|
| **Max_disp** | `run_sensitivity_max_disp.sh` | window_size=17 | max_disp=32,64,96,128 | 视差范围的影响 |
| **Window_size** | `run_sensitivity_window_size.sh` | max_disp=128 | window_size=5,9,13,17,21 | 窗口大小的影响 |
| **Block大小** | `run_sensitivity_block_size.sh` | max_disp=128, window_size=17 | block=(8,8), (16,16), (16,32), (32,16) | 仅测试V5 |
| **多数据集** | `run_multi_dataset.sh` | max_disp=128, window_size=17 | Tsukuba, Teddy, ConesH, Art | 通用性验证 |

---

## 🚀 快速开始（按顺序执行）

### 步骤 1：代码验证
```bash
# 确保串行版、CUDA V1/V2/V5都能运行
gcc 1serial.c pgm_io.c -O2 -o serial
nvcc cuda_v1.cu pgm_io.c -O2 -o cuda_v1
nvcc cuda_v2.cu pgm_io.c -O2 -o cuda_v2
nvcc cuda_v5.cu pgm_io.c -O2 -arch=sm_89 -o cuda_v5

# 快速测试（conesH数据集）
./serial conesH/left.pgm conesH/right.pgm /tmp/test_serial.pgm 128 17
./cuda_v1 conesH/left.pgm conesH/right.pgm /tmp/test_v1.pgm 128 17
./cuda_v2 conesH/left.pgm conesH/right.pgm /tmp/test_v2.pgm 128 17
./cuda_v5 conesH/left.pgm conesH/right.pgm /tmp/test_v5.pgm 128 17

# 验证输出结果一致性
```

### 步骤 2：Max_disp 敏感性实验
```bash
chmod +x run_sensitivity_max_disp.sh
./run_sensitivity_max_disp.sh

# 输出：results/sensitivity_max_disp_TIMESTAMP/summary.csv

# 分析结果
python3 analysis.py results/sensitivity_max_disp_TIMESTAMP/summary.csv
```

**预期结果**：
- V2 约 3 倍快于 V1
- V5 约 1.5 倍快于 V1
- 时间随 max_disp 的线性增长

### 步骤 3：Window_size 敏感性实验
```bash
chmod +x run_sensitivity_window_size.sh
./run_sensitivity_window_size.sh

# 分析结果
python3 analysis.py results/sensitivity_window_size_TIMESTAMP/summary.csv
```

**预期结果**：
- 时间随窗口面积的平方增长
- window_size=5 到 21：面积增长 441/25 = 17.6 倍
- 时间应该增长约 17.6 倍左右

### 步骤 4：Block 大小实验（CUDA V5 only）
```bash
chmod +x run_sensitivity_block_size.sh
./run_sensitivity_block_size.sh

# 分析结果
python3 analysis.py results/sensitivity_block_size_TIMESTAMP/summary.csv
```

**预期结果**：
- 不同 block 大小性能差异通常 < 5%
- RTX 4090 occupancy 高，所以差异不明显
- 记录最优配置用于论文

### 步骤 5：多数据集实验（最后做）

#### 5a. 准备数据集（一次性操作）
```bash
# 创建数据集目录
mkdir -p datasets/{tsukuba_stereo,teddy_stereo,art_stereo,conesH}

# 下载 Middlebury 数据
# https://vision.middlebury.edu/stereo/data/
# - Tsukuba (2001): frame00, im0.pgm + im1.pgm (384×288)
# - Teddy (2001): frame10, im2.pgm + im6.pgm (640×480)
# - Art (2003): frame00, im0.pgm + im1.pgm (1024×768)
# - Cones (2003): im2.pgm + im6.pgm (900×750) ← 已有

# 若图像为PNG格式，需要转换为灰度PGM（可选）
# python3 convert_to_pgm.py datasets/
```

#### 5b. 运行多数据集实验
```bash
chmod +x run_multi_dataset.sh
./run_multi_dataset.sh

# 分析结果
python3 analysis.py results/multi_dataset_TIMESTAMP/summary.csv
```

**预期结果**：
- 加速比在不同数据集上保持一致
- 大图像 (1024×768) 绝对时间更长
- V2 始终最快

---

## 📊 后续分析与可视化

### 用 Excel/LibreOffice 生成图表
```bash
# 将 CSV 文件导入 Excel
# 创建图表：
# 1. 折线图：max_disp vs 时间 (三条线：V1/V2/V5)
# 2. 折线图：window_size vs 时间 (三条线：V1/V2/V5)
# 3. 柱状图：block_size vs 时间 (V5 only)
# 4. 分组柱状图：不同数据集的加速比对比
```

### 用 Python 生成报告
```bash
# 见 analysis.py 中的各个函数，可自行扩展图表生成功能
python3 analysis.py <csv_file>
```

---

## 🎯 实验指标定义

### 1. 运行时间（Time）
- 单位：毫秒（ms）
- 仅计算 GPU kernel 执行时间
- 不包括：内存拷贝、I/O

### 2. 加速比（Speedup）
```
Speedup(V2) = Time(V1) / Time(V2)
Speedup(V5) = Time(V1) / Time(V5)
```
- 以 V1 为基线
- V2 预期 ~3x
- V5 预期 ~1.5x

### 3. 标准差（Std Dev）
- 5 次运行的标准差
- 衡量实验的稳定性
- 应该 < 5% (稳定)

---

## ⚠️ 常见问题

### Q: 某个脚本编译失败？
**A:** 检查 CUDA 版本（需要 12.x）和 GPU 架构（-arch=sm_89）
```bash
nvcc --version
nvidia-smi  # 检查 GPU
```

### Q: 结果波动很大？
**A:** 
- 确保系统空闲（关闭其他应用）
- 增加运行次数（将 RUNS=5 改成 10）
- 对 GPU 做 warm-up（先运行一次整个脚本）

### Q: 多数据集脚本报错"File not found"？
**A:** 检查数据集路径是否正确，按照"步骤 5a" 准备数据集

### Q: 如何从 PNG 转换为灰度 PGM？
```bash
# 需要 ImageMagick
convert input.png -type Grayscale output.pgm

# 或用 Python
python3 << 'EOF'
from PIL import Image
import numpy as np

def png_to_pgm(png_file, pgm_file):
    img = Image.open(png_file).convert('L')  # 转灰度
    img.save(pgm_file)

png_to_pgm('input.png', 'output.pgm')
EOF
```

---

## 📝 论文可用的实验数据

完成上述所有实验后，你将拥有：

1. **表格**：max_disp 参数敏感性（4×3 的表格）
2. **表格**：window_size 参数敏感性（5×3 的表格）
3. **表格**：block 大小对 V5 的影响（4×2 的表格）
4. **表格**：多数据集加速比对比（4 个数据集×3 个版本）
5. **图表**：时间曲线（3 张折线图）
6. **图表**：加速比对比（1 张分组柱状图）

---

## 🎓 建议的实验报告结构

```
CUDA 优化实验报告
│
├─ 实验 1：Max_disp 敏感性
│  ├─ 实验设置
│  ├─ 结果表格
│  ├─ 结果图表（折线图）
│  └─ 分析结论
│
├─ 实验 2：Window_size 敏感性
│  └─ [同上]
│
├─ 实验 3：Block 大小优化
│  └─ [同上]
│
├─ 实验 4：多数据集验证
│  └─ [同上]
│
└─ 综合结论
   ├─ V2 最优版本推荐配置
   ├─ 加速比总结
   └─ 性能瓶颈分析
```

---

## 🔄 迭代建议

1. **第一周**：完成实验 1-3（参数敏感性）
2. **第二周**：完成实验 4（多数据集）+ 数据分析与可视化
3. **第三周**：准备论文、优化展示方式

---

## 📞 如有问题

- 单个脚本执行失败 → 检查编译错误
- 结果异常 → 增加 RUNS 次数重新运行
- 需要新增实验 → 参考脚本模板修改

祝实验顺利！🎉

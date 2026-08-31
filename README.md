# SCSegamba

SCSegamba 是一个用于结构裂缝二值分割的轻量级模型。本仓库当前版本已经支持在不改动模型结构的情况下，通过命令行参数指定不同的裂缝数据集进行训练和测试。

> 当前默认数据集根目录为 `./Dataset/TUT`。如果要训练其他数据集，只需要修改 `--dataset_path`。

---

## 1. 项目结构

```text
SCSegamba/
├── main.py                         # 训练入口：训练、每轮测试、指标评估、保存 checkpoint
├── test.py                         # 单独推理/测试入口：加载指定 checkpoint 输出预测结果
├── eval_compute.py                 # 统计模型 FLOPs 和参数量
├── engine.py                       # 单轮训练逻辑
├── benchmark_trapezoidal.py        # Euler vs trapezoidal_fixed 速度对比
├── datasets/
│   ├── __init__.py                 # 数据集工厂，根据 --dataset_mode 创建 dataloader
│   ├── base_dataset.py             # 数据集基类
│   ├── crack_dataset.py            # 裂缝分割数据集读取逻辑
│   ├── image_folder.py             # 递归读取图像文件
│   └── utils.py                    # mask 转 tensor 等工具
├── models/
│   ├── __init__.py                 # build_model 入口
│   ├── decoder.py                  # SAVSS backbone + MFS decoder + loss
│   ├── MFS.py                      # 多尺度特征融合模块
│   ├── GBC.py                      # Gated Bottleneck Convolution 相关模块
│   ├── DySample.py                 # 动态上采样模块
│   └── PAF.py                      # 特征对齐/融合模块
├── mmcls/SAVSS_dev/models/SAVSS/
│   ├── SAVSS.py                    # SAVSS backbone 主体
│   ├── SAVSS_layer.py              # SAVSS layer、SASS 扫描、selective scan 路由
│   ├── selective_scan_trapezoidal.py       # 纯 PyTorch 指数-梯形 selective scan 参考实现
│   └── selective_scan_trapezoidal_triton.py # Triton 融合指数-梯形 selective scan
├── tests/
│   ├── test_SAVSS_trapezoidal.py   # SAVSS / 梯形离散化单元测试
│   └── test_triton_trapezoidal_gradients.py # Triton 梯度快速对比脚本
├── eval/
│   ├── evaluate.py                 # mIoU、ODS、OIS、F1、Precision、Recall 计算
│   └── data_io.py                  # 评估图像读取工具
├── requirements-rtx50.txt          # RTX 50 系 / CUDA 12.8 环境依赖
└── requirements-legacy-cu116.txt   # 旧 CUDA 11.6 环境依赖
```

---

## 2. 模型结构说明

当前训练入口通过 `models.build_model(args)` 构建模型，实际调用链为：

```text
main.py
└── models/build_model(args)
    └── models/decoder.py::build(args)
        ├── SAVSS backbone
        ├── MFS decoder
        └── BCE + Dice 组合损失
```

整体结构如下：

```text
Input image: B × 3 × 512 × 512
        │
        ▼
SAVSS Backbone
        │
        ├── c4: B × 128 × 64  × 64
        ├── c3: B × 64  × 128 × 128
        ├── c2: B × 32  × 256 × 256
        └── c1: B × 16  × 512 × 512
        │
        ▼
MFS Multi-scale Feature Fusion Decoder
        │
        ▼
Output logits: B × 1 × 512 × 512
```

### 2.1 SAVSS Backbone

Backbone 定义在 `mmcls/SAVSS_dev/models/SAVSS/SAVSS.py` 中，训练时使用：

```python
SAVSS(
    arch='Crack',
    out_indices=(0, 1, 2, 3),
    drop_path_rate=0.2,
    final_norm=True,
    convert_syncbn=True,
)
```

`arch='Crack'` 的主要配置：

| 配置项 | 数值 |
|---|---:|
| `patch_size` | 8 |
| `embed_dims` | 256 |
| `num_layers` | 4 |
| `num_convs_patch_embed` | 2 |
| `d_state` | 16 |
| `expand` | 2 |
| `conv_size` | 7 |
| `default_hw_shape` | 64 × 64 |
| `use_rms_norm` | False |

Backbone 的主要流程：

1. **ConvPatchEmbed**  
   将输入图像划分为 patch token。默认输入为 `512 × 512`，`patch_size=8`，因此初始 token 空间尺寸为 `64 × 64`。

2. **Position Embedding**  
   对 patch token 添加位置编码。

3. **4 层 SAVSS_Layer**  
   每一层中包含 Mamba selective scan 相关计算，并结合结构感知扫描策略 SASS（Structure-Aware Scanning Strategy）。SASS 会以多种方向/路径扫描二维 token，以增强裂缝细长结构和连续拓扑信息的建模能力。

4. **多层特征输出**  
   `out_indices=(0, 1, 2, 3)` 表示输出 4 个阶段的特征。代码中会将 256 维 token 特征分别转换为：
   - `128` 通道，恢复到 `64 × 64`
   - `64` 通道，恢复到 `128 × 128`
   - `32` 通道，恢复到 `256 × 256`
   - `16` 通道，恢复到 `512 × 512`

### 2.2 SAVSS Layer 与 SASS

`SAVSS_layer.py` 中的核心模块是 `SAVSS_2D`。其主要思想是：

1. 将二维图像 token 按结构感知路径展开。
2. 使用多方向扫描顺序建模裂缝像素之间的连续关系。
3. 根据配置调用 selective scan 函数进行状态空间递推。
4. 结合轻量卷积模块补充局部形态信息。

`SAVSS_2D` 支持两种离散化方式，通过 `discretization` 参数选择：

- `euler`（默认）：使用 `mamba_ssm.ops.selective_scan_interface.selective_scan_fn` 的一阶指数-Euler 离散化核函数。
- `trapezoidal_fixed`：使用本项目新增的 **Triton 融合核函数** 实现指数-梯形离散化，可通过 `--discretization trapezoidal_fixed` 开启。

`trapezoidal_fixed` 在加载时会优先尝试编译 Triton kernel；如果 Triton 不可用或编译失败，会自动回退到纯 PyTorch 实现，不会中断训练。RTX 50 系显卡需要 CUDA 12.8+ 版本的 PyTorch、对应编译好的 Mamba 扩展，以及可用的 Triton 环境。

### 2.3 离散化方式说明

`SAVSS_2D` 的状态空间离散化支持两种方式：

#### `euler`（默认）

对应标准 S6 selective scan 的指数-Euler 离散化：

```text
h_t = exp(Δ_t A) h_{t-1} + Δ_t B_t x_t
y_t = C_t^T h_t + D x_t
```

- 调用 `mamba_ssm` 的 fused CUDA kernel。
- 速度与原始 Mamba S6 一致。

#### `trapezoidal_fixed`

对应指数-梯形（exponential-trapezoidal）离散化，使用固定权重 `λ=0.5`：

```text
h_t = exp(Δ_t A) h_{t-1}
      + (1-λ) Δ_t exp(Δ_t A) B_{t-1} x_{t-1}
      + λ Δ_t B_t x_t
y_t = C_t^T h_t + D x_t
```

- `t=0` 时没有前一项，退化为 Euler 第一步 `h_0 = Δ_0 B_0 x_0`。
- `λ=1.0` 时严格等价于 Euler 离散化，可用于一致性验证。
- 本项目为该离散化实现了 Triton 融合核函数（`mmcls/SAVSS_dev/models/SAVSS/selective_scan_trapezoidal_triton.py`），前向与反向均在 CUDA 上完成，无需手动写 C++ extension。
- 若 Triton 编译失败，自动回退到同目录下的纯 PyTorch 参考实现（`selective_scan_trapezoidal.py`）。

性能参考（RTX 5080，单 batch，输入 512×512）：

```text
Euler (mamba_ssm fused)       : ~0.104 s / iter
trapezoidal_fixed (Triton)    : ~0.100 s / iter
```

> 注：不同 batch size 与序列长度下速度可能略有差异；梯形路径不会显著慢于 Euler 路径。

---

### 2.4 GBC：Gated Bottleneck Convolution

`models/GBC.py` 中定义了两个核心模块：

- `BottConv`
- `GBC`

`BottConv` 是轻量瓶颈卷积：

```text
1×1 pointwise conv
→ depthwise conv
→ 1×1 pointwise conv
```

`GBC` 通过多分支瓶颈卷积和门控乘法增强裂缝形态特征：

```text
x ──────────────── residual ──────────────┐
│                                          │
├─ BottConv + Norm + ReLU                  │
├─ BottConv + Norm + ReLU  ── x1           │
│                                          ├─ x1 * x2 → BottConv → + residual
└─ BottConv + Norm + ReLU  ── x2           │
                                           │
```

GBC 在 backbone 和 decoder 中都会使用，用来增强裂缝边缘、细线和形态结构信息。

### 2.5 MFS Decoder

Decoder 定义在 `models/MFS.py` 中。MFS 表示多尺度特征融合模块。

Backbone 输出的四个尺度特征会经过以下步骤：

1. **通道映射**  
   使用线性层将不同通道数统一投影到 `embedding_dim=8`：
   - `128 → 8`
   - `64 → 8`
   - `32 → 8`
   - `16 → 8`

2. **动态上采样 DySample**  
   将低分辨率特征恢复到 `512 × 512`：
   - `c4` 上采样 `8×`
   - `c3` 上采样 `4×`
   - `c2` 上采样 `2×`
   - `c1` 保持原尺寸

3. **特征拼接和融合**  
   四个尺度特征在通道维拼接，得到 `32` 通道特征，再经过：
   - `GBC`
   - `BottConv`
   - `Dropout`
   - `1×1 Conv`

4. **输出预测图**  
   最终输出为单通道 logits：

```text
B × 1 × 512 × 512
```

注意：模型输出是 logits，训练损失内部会对 Dice 部分使用 `sigmoid`。

### 2.6 损失函数

损失函数定义在 `models/decoder.py` 中，为 BCEWithLogitsLoss 和 DiceLoss 的加权组合：

```text
loss = BCELoss_ratio × BCEWithLogitsLoss + DiceLoss_ratio × DiceLoss
```

默认权重：

| 参数 | 默认值 |
|---|---:|
| `--BCELoss_ratio` | 0.83 |
| `--DiceLoss_ratio` | 0.17 |

---

## 3. 环境配置

### 3.1 使用已经创建好的 Scs 环境

如果已经创建好环境，直接激活：

```bash
conda activate Scs
```

建议先检查 GPU 和 PyTorch：

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda runtime:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
    print('capability:', torch.cuda.get_device_capability(0))
PY
```

### 3.2 RTX 50 系显卡环境说明

RTX 50 系 / Blackwell GPU 需要较新的 NVIDIA 驱动、CUDA 12.8+ PyTorch，以及匹配当前 CUDA/PyTorch 的 `mamba-ssm` 和 `causal-conv1d`。如果计划使用 `trapezoidal_fixed` 离散化，还需要可用的 Triton 环境（`requirements-rtx50.txt` 中已包含）。

推荐依赖安装方式：

```bash
conda create -n Scs python=3.10 -y
conda activate Scs

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-rtx50.txt
```

如果 `mamba-ssm` 或 `causal-conv1d` 需要从源码编译，RTX 50 系建议设置：

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
pip install --no-cache-dir --no-build-isolation causal-conv1d mamba-ssm
```

验证 Mamba selective scan：

```bash
python - <<'PY'
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
print('mamba selective_scan OK')
PY
```

验证 Triton（用于 `trapezoidal_fixed` 加速）：

```bash
python - <<'PY'
import triton
print('triton:', triton.__version__)
from mmcls.SAVSS_dev.models.SAVSS.selective_scan_trapezoidal_triton import selective_scan_trapezoidal_triton_fn
print('trapezoidal Triton kernel OK')
PY
```

---

## 4. 数据集格式

当前数据加载器 `datasets/crack_dataset.py` 支持 `train_img/train_lab/val_img/val_lab/test_img/test_lab` 这一套统一格式。

### 4.1 推荐格式：统一转换后的数据集

推荐使用仓库内 `./Dataset/<dataset_name>` 目录下的统一格式：

```text
Dataset/<dataset_name>/
├── train_img/
├── train_lab/
├── val_img/
├── val_lab/
├── test_img/
└── test_lab/
```

训练时使用 `train` split，每个 epoch 结束后使用 `test` split 做测试和评估。

图像和 mask 需要同名，例如：

```text
train_img/example_001.jpg
train_lab/example_001.png
```

mask 默认按 `.png` 查找。

### 4.2 兼容旧格式

为了兼容旧的两目录命名，也支持：

```text
<dataset_path>/train_img/
<dataset_path>/train_lab/
<dataset_path>/test_img/
<dataset_path>/test_lab/
```

如果数据集还带有验证集，也可以继续保留：

```text
<dataset_path>/val_img/
<dataset_path>/val_lab/
```

### 4.3 当前可用数据集

当前 `./Dataset` 下可直接尝试训练的数据集包括：

```text
CRACK500
CRKWH100
CrackLS315
CrackTree260
Deepcrack
EdmCrack600
Gaps384
Noname
Stone331
TUT
```

`CrackForest` 当前转换结果不完整，不建议直接训练；如果目录中没有 `train_img`、`train_lab` 等文件夹，程序会直接报错提示找不到 split。

---

## 5. 训练

训练入口是 `main.py`。

### 5.1 训练 TUT

```bash
cd /home/xby/SCSegamba
conda activate Scs

python main.py \
  --dataset_path ./Dataset/TUT \
  --dataset_mode crack \
  --device cuda \
  --epochs 50 \
  --batch_size_train 1 \
  --batch_size_test 1 \
  --num_threads 1
```

### 5.2 训练 CRACK500

```bash
cd /home/xby/SCSegamba
conda activate Scs

python main.py \
  --dataset_path ./Dataset/CRACK500 \
  --dataset_mode crack \
  --device cuda \
  --epochs 50 \
  --batch_size_train 1 \
  --batch_size_test 1 \
  --num_threads 1
```

### 5.3 训练 Deepcrack

```bash
cd /home/xby/SCSegamba
conda activate Scs

python main.py \
  --dataset_path ./Dataset/Deepcrack \
  --dataset_mode crack \
  --device cuda \
  --epochs 50 \
  --batch_size_train 1 \
  --batch_size_test 1 \
  --num_threads 1
```

### 5.4 使用 trapezoidal_fixed 离散化训练

默认使用 `euler` 离散化。如果需要开启指数-梯形离散化，在训练命令中加入 `--discretization trapezoidal_fixed`：

```bash
cd /home/xby/SCSegamba
conda activate Scs

python main.py \
  --dataset_path ./Dataset/TUT \
  --dataset_mode crack \
  --device cuda \
  --discretization trapezoidal_fixed \
  --epochs 50 \
  --batch_size_train 1 \
  --batch_size_test 1 \
  --num_threads 1
```

> 注：`trapezoidal_fixed` 会优先使用 Triton 融合核函数。如果 Triton 在当前环境无法编译，会自动回退到纯 PyTorch 实现。

### 5.5 从断点恢复训练

如果训练中断，可以使用 `--resume` 从已有 checkpoint 继续训练：

```bash
cd /home/xby/SCSegamba
conda activate Scs

python main.py \
  --dataset_path ./Dataset/TUT \
  --dataset_mode crack \
  --device cuda \
  --resume ./checkpoints/weights/<训练输出目录>/checkpoint.pth \
  --epochs 50 \
  --batch_size_train 1 \
  --batch_size_test 1 \
  --num_threads 1
```

也可以把 `checkpoint.pth` 换成某个历史轮次的 `checkpoint{epoch}.pth`。恢复训练后，程序会自动加载：

- 模型权重
- optimizer 状态
- lr scheduler 状态
- 上次保存的 epoch



| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--dataset_path` | `./Dataset/TUT` | 数据集根目录 |
| `--dataset_mode` | `crack` | 数据集类型，当前裂缝分割使用 `crack` |
| `--device` | `cuda` | 训练设备，可用 `cuda`、`cuda:0`、`cpu` |
| `--discretization` | `euler` | selective scan 离散化方式，可选 `euler`、`trapezoidal_fixed` |
| `--epochs` | 50 | 训练轮数 |
| `--batch_size_train` | 1 | 训练 batch size |
| `--batch_size_test` | 1 | 测试 batch size |
| `--num_threads` | 1 | DataLoader worker 数 |
| `--lr` | `5e-4` | 初始学习率 |
| `--min_lr` | `1e-6` | PolyLR 最小学习率 |
| `--weight_decay` | `0.01` | 权重衰减 |
| `--lr_scheduler` | `PolyLR` | 学习率策略，可选 `PolyLR`、`StepLR`、`CosLR` |
| `--load_width` | 512 | 输入 resize 宽度 |
| `--load_height` | 512 | 输入 resize 高度 |
| `--BCELoss_ratio` | 0.83 | BCE loss 权重 |
| `--DiceLoss_ratio` | 0.17 | Dice loss 权重 |
| `--output_dir` | `./checkpoints/weights` | 权重保存目录 |

---

## 7. 训练过程输出

运行 `main.py` 后，每个 epoch 会执行：

```text
train → test → evaluate → save checkpoint
```

### 7.1 checkpoint 保存位置

训练输出目录格式：

```text
checkpoints/weights/<时间>_Dataset-><数据集名>/
```

例如：

```text
checkpoints/weights/2026_08_17_12:00:00_Dataset->TUT/
├── checkpoint.pth
├── checkpoint0.pth
├── checkpoint1.pth
├── ...
└── checkpoint_best.pth
```

说明：

- `checkpoint.pth`：当前最新 checkpoint
- `checkpoint{epoch}.pth`：每一轮保存的 checkpoint
- `checkpoint_best.pth`：根据当前评估结果中最高 `mIoU` 保存的最好模型

### 7.2 预测结果保存位置

每个 epoch 的测试预测会保存到：

```text
results/<时间>_Dataset-><数据集名>/results_<epoch>/
```

每张图会生成两类文件：

```text
xxx_lab.png   # ground truth mask
xxx_pre.png   # prediction mask
```

### 7.3 日志保存位置

训练日志保存在：

```text
checkpoints/<时间>_Dataset-><数据集名>/
```

其中包含 train、test、eval 相关日志。

---

## 8. 单独测试 / 推理

测试入口是 `test.py`。需要手动指定数据集和 checkpoint。

```bash
cd /home/xby/SCSegamba
conda activate Scs

python test.py \
  --dataset_path ./Dataset/TUT \
  --dataset_mode crack \
  --checkpoint_path ./checkpoints/weights/<训练输出目录>/checkpoint_best.pth \
  --device cuda
```

例如：

```bash
python test.py \
  --dataset_path ./Dataset/CRACK500 \
  --dataset_mode crack \
  --checkpoint_path ./checkpoints/weights/2026_08_17_12:00:00_Dataset->CRACK500/checkpoint_best.pth \
  --device cuda
```

测试结果默认保存到：

```text
results/results_test/<checkpoint文件名>/
```

例如使用 `checkpoint_best.pth` 时：

```text
results/results_test/checkpoint_best/
```

---

## 9. 评估指标

训练时每个 epoch 后会自动调用 `eval/evaluate.py` 中的 `eval()` 函数，对当前 epoch 的测试结果进行评估。

主要指标包括：

| 指标 | 含义 |
|---|---|
| `mIoU` | mean Intersection over Union |
| `ODS` | Optimal Dataset Scale |
| `OIS` | Optimal Image Scale |
| `F1` | F1-score |
| `Precision` | 精确率 |
| `Recall` | 召回率 |

训练过程中会根据 `mIoU` 更新并保存 `checkpoint_best.pth`。

如果要对已有结果目录手动评估，可根据 `eval/evaluate.py` 中的路径修改 `results_dir`，或者在 Python 中直接调用：

```python
from eval.evaluate import eval

metrics = eval(log_eval, results_dir, epoch)
print(metrics)
```

其中 `results_dir` 目录需要包含成对的：

```text
xxx_lab.png
xxx_pre.png
```

---

## 10. 统计 FLOPs 和参数量

使用 `eval_compute.py`：

```bash
cd /home/xby/SCSegamba
conda activate Scs

python eval_compute.py --device cuda
```

该脚本会构造一个 `1 × 3 × 512 × 512` 的输入，并输出：

```text
flops(G): ... params(M): ...
```

---

## 11. 快速验证命令

### 11.1 检查 Python 文件语法

```bash
cd /home/xby/SCSegamba
conda activate Scs

python -m py_compile main.py test.py datasets/crack_dataset.py
```

### 11.2 检查数据集能否正常读取

```bash
cd /home/xby/SCSegamba
conda activate Scs

python - <<'PY'
import argparse
from main import get_args_parser
from datasets import create_dataset

parser = argparse.ArgumentParser(parents=[get_args_parser()])
args = parser.parse_args([])
args.dataset_path = './Dataset/TUT'
args.dataset_mode = 'crack'
args.batch_size = 1
args.num_threads = 0
args.serial_batches = True

for phase in ['train', 'test']:
    args.phase = phase
    dl = create_dataset(args)
    batch = next(iter(dl))
    print(phase, len(dl), batch['image'].shape, batch['label'].shape)
PY
```

正常情况下会看到类似：

```text
train 987 torch.Size([1, 3, 512, 512]) torch.Size([1, 1, 512, 512])
test 282 torch.Size([1, 3, 512, 512]) torch.Size([1, 1, 512, 512])
```

### 11.3 进行 1 个 epoch 的训练 smoke test

```bash
cd /home/xby/SCSegamba
conda activate Scs

python main.py \
  --dataset_path ./Dataset/TUT \
  --dataset_mode crack \
  --device cuda \
  --epochs 1 \
  --batch_size_train 1 \
  --batch_size_test 1 \
  --num_threads 1
```

### 11.4 验证 trapezoidal Triton kernel

运行单元测试，检查 Triton 实现与纯 PyTorch 参考实现的前向、反向一致性：

```bash
cd /home/xby/SCSegamba
conda activate Scs

python -m pytest tests/test_SAVSS_trapezoidal.py -v
```

如果只想快速对比梯度：

```bash
python tests/test_triton_trapezoidal_gradients.py
```

---

## 12. 常见问题

### 12.1 找不到 split 目录

如果报错类似：

```text
Could not find split 'train' under '...'
```

说明数据集目录不符合以下任一格式：

```text
<dataset_path>/train_img/
<dataset_path>/train_lab/
<dataset_path>/test_img/
<dataset_path>/test_lab/
```

或旧格式：

```text
<dataset_path>/train_img
<dataset_path>/train_lab
```

请检查 `--dataset_path` 是否指向具体数据集目录，而不是总目录。例如应该写：

```bash
--dataset_path ./Dataset/TUT
```

不要写成：

```bash
--dataset_path ./Dataset
```

### 12.2 mask 读取失败

如果报错：

```text
Could not read mask: ...
```

请检查图像和 mask 是否同名。当前读取逻辑默认：

```text
<dataset_path>/train_img/example_001.jpg
<dataset_path>/train_lab/example_001.png
```

### 12.3 CUDA 或 Mamba 报错

如果在 RTX 50 系显卡上出现 CUDA / Mamba 相关错误，请检查：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
PY
```

RTX 50 系建议使用 CUDA 12.8+ PyTorch，并重新编译 `mamba-ssm`、`causal-conv1d`：

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
pip install --no-cache-dir --no-build-isolation causal-conv1d mamba-ssm
```

若在使用 `--discretization trapezoidal_fixed` 时遇到 Triton 编译错误，`SAVSS_2D` 会自动回退到纯 PyTorch 实现继续训练。

### 12.4 显存不足

可以先降低 batch size：

```bash
--batch_size_train 1 --batch_size_test 1
```

如果仍然不足，可以尝试使用更小输入尺寸：

```bash
--load_width 384 --load_height 384
```

但需要注意，改变输入尺寸可能影响结果可比性。

---

## 13. 推荐工作流

完整训练一个数据集推荐流程：

```bash
cd /home/xby/SCSegamba
conda activate Scs

# 1. 检查环境
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY

# 2. 检查数据读取
python - <<'PY'
import argparse
from main import get_args_parser
from datasets import create_dataset
parser = argparse.ArgumentParser(parents=[get_args_parser()])
args = parser.parse_args([])
args.dataset_path = './Dataset/TUT'
args.dataset_mode = 'crack'
args.phase = 'train'
args.batch_size = 1
args.num_threads = 0
dl = create_dataset(args)
batch = next(iter(dl))
print(len(dl), batch['image'].shape, batch['label'].shape)
PY

# 3. 正式训练
python main.py \
  --dataset_path ./Dataset/TUT \
  --dataset_mode crack \
  --device cuda \
  --epochs 50 \
  --batch_size_train 1 \
  --batch_size_test 1 \
  --num_threads 1

# 4. 使用最佳 checkpoint 单独测试
python test.py \
  --dataset_path ./Dataset/TUT \
  --dataset_mode crack \
  --checkpoint_path ./checkpoints/weights/<训练输出目录>/checkpoint_best.pth \
  --device cuda
```

# Enhancing Binary Neural Networks with Hybrid-Granularity Contrastive Deep Supervision

> **Official implementation** of the paper submitted to *The Visual Computer*.

This repository contains the official implementation of **HG-CDS** (**H**ybrid-**G**ranularity **C**ontrastive **D**eep **S**upervision), a training framework that applies contrastive supervision at multiple feature granularities to improve Binary Neural Network (BNN) training.

HG-CDS is evaluated on two architectures:
- **CNN**: Binary ResNet on CIFAR-10 / ImageNet-1K
- **ViT (DeiT)**: BinaryViT on CIFAR-10 / ImageNet-1K

---

## Method Overview

HG-CDS introduces **Hybrid-Granularity Contrastive Deep Supervision** at intermediate layers of a BNN:

1. **Multi-stage Auxiliary Classifiers**: Lightweight depthwise separable convolution heads (`SepConv`) are attached to intermediate feature maps at each stage, extracting features at different spatial granularities (coarse-to-fine).
2. **Supervised Contrastive Loss (SupCon)**: Applied at each auxiliary output using two augmented views of the same input. Pulls together same-class embeddings and pushes apart different-class embeddings across all granularity levels.
3. **Knowledge Distillation (KD)**: A full-precision teacher model optionally provides soft-label supervision (KL-divergence) and feature-level L2 alignment at each stage, further bridging the representational gap.

The total training objective combines:
- **CE loss**: standard cross-entropy on the final output
- **CL loss** (`--cl`): weighted sum of SupCon losses over all auxiliary stages
- **KD loss** (`--kd`): weighted sum of soft-label KL distillation + per-stage feature L2 alignment

For **BinaryViT**, auxiliary classifiers (`ViTAuxiliaryClassifier`) convert ViT's sequence-format features `[B, N, D]` into spatial format via `ViTSepConv` and progressively downsample them to a unified 512-dim embedding for contrastive learning.

---

## Requirements

### CNN (Binary ResNet)

```bash
pip install -r CNN/requirements.txt
```

Key dependencies:
- Python 3.8
- PyTorch >= 1.10
- torchvision >= 0.11
- tensorboardX >= 2.6
- pandas >= 2.0
- Pillow >= 10.0
- matplotlib >= 3.7

### DeiT (BinaryViT)

```bash
pip install -r DeiT/requirements.txt
```

Key dependencies:
- Python 3.8
- PyTorch >= 1.10.1
- torchvision >= 0.11.2
- timm == 0.6.12
- transformers >= 4.20.1

---

## Training

### CNN on CIFAR-10 (with CDS + Teacher KD)

```bash
cd CNN
python distill.py \
    --gpus 0 \
    --model resnet18_1w1a \
    --model_teacher resnet18_1w1a \
    --teacher_path /path/to/teacher/checkpoint \
    --dataset cifar10 \
    --data_path /path/to/cifar10 \
    --epochs 1200 \
    --lr 0.1 \
    -b 512 -bt 128 \
    --cl 2 --kd 1 \
    --warm_up \
    --results_dir ./results \
    --save run_cifar10
```

Or use the provided script (set `DATA_PATH` to your dataset directory):
```bash
cd CNN
DATA_PATH=/path/to/cifar10 bash run_cifar.sh
```

### CNN on ImageNet-1K (with CDS)

```bash
cd CNN
bash run_imagenet.sh
```

### BinaryViT on CIFAR-10 (with CDS + Teacher KD)

**Step 1:** Train or download the full-precision DeiT-S teacher:
```bash
cd DeiT
DATA_DIR=/path/to/datasets bash scripts/run_deit-small-patch16-224.sh
```

**Step 2:** Train BinaryViT with CDS:
```bash
cd DeiT
DATA_DIR=/path/to/datasets bash scripts/run_binaryvit-small-patch4-224-cifar10.sh
```

Or manually:
```bash
cd DeiT
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py \
    --batch-size=64 \
    --epochs=300 \
    --data-set=CIFAR10 \
    --data-path=/path/to/datasets \
    --model=configs/binaryvit-small-patch4-224-cifar10 \
    --model-type=extra-res-pyramid \
    --teacher-model-type=deit \
    --teacher-model=configs/deit-small-patch16-224-cifar10 \
    --teacher-model-file=logs/deit-small-patch16-224-cifar10-teacher/best.pth \
    --weight-bits=1 --input-bits=1 \
    --replace-ln-bn \
    --avg-res3 --avg-res5 \
    --output-dir=logs/binaryvit_cifar10_cds
```

### BinaryViT on ImageNet-1K (with CDS + Teacher KD)

**Step 1:** Train the full-precision DeiT-S teacher:
```bash
cd DeiT
DATA_DIR=/path/to/imagenet bash scripts/run_deit-small-patch16-224.sh
```

**Step 2:** Train BinaryViT with CDS:
```bash
cd DeiT
DATA_DIR=/path/to/imagenet bash scripts/run_binaryvit-small-patch4-224.sh
```

---

## Key Arguments

### CNN (`distill.py`)

| Argument | Default | Description |
|---|---|---|
| `--model` | `resnet18_1w1a` | Binary student model |
| `--model_teacher` | `resnet18_1w1a` | Teacher model architecture |
| `--teacher_path` | — | Path to teacher checkpoint directory |
| `--dataset` | `cifar10` | Dataset: `cifar10`, `cifar100`, `imagenet` |
| `--data_path` | — | Path to dataset |
| `--epochs` | `600` | Training epochs |
| `--lr` | `0.1` | Learning rate |
| `--cl` | `1.0` | Contrastive loss weight |
| `--kd` | `1.0` | Knowledge distillation loss weight |
| `--temp` | `4.0` | Distillation temperature |
| `--supervision` | `True` | Use supervised (SupCon) vs. unsupervised (SimCLR) contrastive loss |
| `--warm_up` | — | Enable learning rate warm-up (5-epoch linear ramp) |
| `--save` | — | Output subdirectory name under `--results_dir` |

### DeiT (`main.py`)

| Argument | Default | Description |
|---|---|---|
| `--model` | — | Path to model config directory |
| `--model-type` | — | Model type: `deit`, `extra-res`, `extra-res-pyramid` |
| `--data-set` | `CIFAR10` | Dataset: `CIFAR10`, `CIFAR100`, `IMNET` |
| `--data-path` | `./data` | Path to dataset |
| `--weight-bits` | `1` | Weight quantization bits |
| `--input-bits` | `1` | Activation quantization bits |
| `--cl` | `0` | Contrastive loss weight (0 = disabled) |
| `--kd` | `1.0` | KD loss coefficient |
| `--temp` | `0.07` | SupConLoss temperature |
| `--teacher-model-type` | — | Teacher model type (e.g., `deit`) |
| `--teacher-model` | — | Path to teacher model config |
| `--teacher-model-file` | — | Path to teacher model weights |
| `--replace-ln-bn` | — | Replace LayerNorm with BatchNorm |
| `--avg-res3` | — | Enable 3×3 average-pooling residual at FFN |
| `--avg-res5` | — | Enable 5×5 average-pooling residual at FFN |
| `--use-multiscale-cds` | — | Enable multi-scale region-token CDS |

---

## Project Structure

```
HG-CDS/
├── CNN/                          # Binary ResNet implementation
│   ├── distill.py                # Main training script (CDS + KD)
│   ├── main_cifar.py             # Training script (CDS only, no KD)
│   ├── main_imagenet.py          # ImageNet training script
│   ├── models_cifar/
│   │   ├── resnet.py             # Binary ResNet-20 for CIFAR
│   │   └── resnet2.py            # Binary ResNet-18/34/50 for CIFAR
│   ├── models_imagenet/
│   │   └── loss.py               # SupConLoss, distillation, CrossEntropy
│   ├── modules/
│   │   └── binarized_modules.py  # BinarizeConv2d, BinarizeLinear
│   ├── dataset/                  # Data loading utilities
│   ├── utils/                    # Logging, checkpointing, options
│   ├── run_cifar.sh              # CIFAR-10 training script
│   └── run_imagenet.sh           # ImageNet training script
│
└── DeiT/                         # BinaryViT implementation
    ├── main.py                   # Main training script
    ├── models.py                 # Model factory and argument parsing
    ├── losses.py                 # DistributionLoss (KL divergence KD)
    ├── engine.py                 # Training/evaluation loop
    ├── datasets.py               # Dataset loading
    ├── transformer/
    │   ├── cds_modules.py        # ViTAuxiliaryClassifier, SepConv, ViTSepConv
    │   ├── modeling_vit_extra_res_pyramid.py  # Pyramidal BinaryViT (main model)
    │   ├── modeling_vit_extra_res.py          # BinaryViT with extra residuals
    │   ├── modeling_vit.py                    # Standard ViT backbone
    │   ├── multi_scale_tokens.py              # Multi-scale token extraction
    │   └── utils_quant.py                     # Quantization utilities
    ├── cds_utils/
    │   └── sup_con_loss.py       # SupConLoss, TwoCropTransform
    ├── configs/                  # Model configuration files
    └── scripts/                  # Training shell scripts
```

---

## Datasets

### CIFAR-10 / CIFAR-100

Downloaded automatically by torchvision when `--data-path` is specified.

### ImageNet-1K

Standard ImageNet directory structure expected:
```
/path/to/imagenet/
├── train/
│   ├── n01440764/
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

---

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{Zefang2026hgcds,
  title={Enhancing Binary Neural Networks with Hybrid-Granularity Contrastive Deep Supervision},
  author={Zefang Wang and Yingqing Yang and Yuhang Dong and Guangyuan Lu and Guanzhong Tian},
  journal={The Visual Computer},
  year={2026},
  note={Under Review}
}
```

# 🔬 Enhancing Binary Neural Networks with Hybrid-Granularity Contrastive Deep Supervision

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18934169.svg)](https://doi.org/10.5281/zenodo.18934169)
![Python](https://img.shields.io/badge/Python-3.8-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.10-orange?logo=pytorch)
![License](https://img.shields.io/github/license/aden9460/HG-CDS)

> 📄 **Official implementation** of the paper submitted to *The Visual Computer*.

This repository contains the official implementation of **HG-CDS** (**H**ybrid-**G**ranularity **C**ontrastive **D**eep **S**upervision), a training framework that applies contrastive supervision at multiple feature granularities to improve Binary Neural Network (BNN) training.

HG-CDS is evaluated on two architectures:
- 🏗️ **CNN**: Binary ResNet on CIFAR-10 / ImageNet-1K
- 🤖 **ViT (DeiT)**: BinaryViT on CIFAR-10 / ImageNet-1K

---

## 💡 Method Overview

HG-CDS introduces **Hybrid-Granularity Contrastive Deep Supervision** at intermediate layers of a BNN:

1. **🔧 Multi-stage Auxiliary Classifiers**: Lightweight depthwise separable convolution heads (`SepConv`) are attached to intermediate feature maps at each stage, extracting features at different spatial granularities (coarse-to-fine).
2. **📐 Supervised Contrastive Loss (SupCon)**: Applied at each auxiliary output using two augmented views of the same input. Pulls together same-class embeddings and pushes apart different-class embeddings across all granularity levels.
3. **🎓 Knowledge Distillation (KD)**: A full-precision teacher model optionally provides soft-label supervision (KL-divergence) and feature-level L2 alignment at each stage, further bridging the representational gap.

The total training objective combines:
- **CE loss**: standard cross-entropy on the final output
- **CL loss** (`--cl`): weighted sum of SupCon losses over all auxiliary stages
- **KD loss** (`--kd`): weighted sum of soft-label KL distillation + per-stage feature L2 alignment

For **BinaryViT**, auxiliary classifiers (`ViTAuxiliaryClassifier`) convert ViT's sequence-format features `[B, N, D]` into spatial format via `ViTSepConv` and progressively downsample them to a unified 512-dim embedding for contrastive learning.

---

## ⚙️ Requirements

### 🏗️ CNN (Binary ResNet)

```bash
pip install -r CNN/requirements.txt
```

Key dependencies:
- Python 3.8
- PyTorch >= 1.10
- torchvision >= 0.11
- tensorboardX >= 2.6

### 🤖 DeiT (BinaryViT)

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

## 🚀 Training

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

## 📝 Citation

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

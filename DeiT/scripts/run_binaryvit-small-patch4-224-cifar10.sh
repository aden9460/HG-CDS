#!/bin/bash
# DATA_DIR: path to the directory containing your dataset (e.g., /path/to/datasets)
DATA_DIR=${DATA_DIR:-./data}

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master-port=29502 --nproc_per_node=4 main.py \
    --num-workers=32 \
    --batch-size=64 \
    --epochs=300 \
    --dropout=0.0 \
    --drop-path=0.0 \
    --opt=adamw \
    --sched=cosine \
    --weight-decay=0.00 \
    --lr=1e-4 \
    --warmup-epochs=0 \
    --color-jitter=0.0 \
    --aa=noaug \
    --reprob=0.0 \
    --mixup=0.0 \
    --data-set=CIFAR10 \
    --cutmix=0.0 \
    --data-path=${DATA_DIR} \
    --output-dir=logs/binaryvit-small-patch4-224-cifar10 \
    --teacher-model-type=deit \
    --teacher-model=configs/deit-small-patch16-224-cifar10 \
    --teacher-model-file=logs/deit-small-patch16-224-cifar10-teacher/best.pth \
    --model=configs/binaryvit-small-patch4-224-cifar10 \
    --model-type=extra-res-pyramid \
    --replace-ln-bn \
    --weight-bits=1 \
    --input-bits=1 \
    --avg-res3 \
    --avg-res5

    # --resume=logs/binaryvit-small-patch4-224-cifar10/checkpoint.pth \
    # --current-best-model=logs/binaryvit-small-patch4-224-cifar10/best.pth \

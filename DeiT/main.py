#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BinaryViT + CDS Training Script (CIFAR-10/100 and ImageNet)
Supports distributed training via DDP.

Features:
1. TwoCropTransform for data augmentation (generates two views per image)
2. Trains BinaryViT + auxiliary classifiers (CDS)
3. Computes classification loss + contrastive learning loss
4. Optional gradient statistics tracking
5. Distributed training support (DDP)
"""

import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import os
from pathlib import Path

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from transformers import ViTConfig
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# BinaryViT modules
from transformer.modeling_vit_extra_res_pyramid import ViTForImageClassification as PyramidViT, generating_stage_per_depth
from transformer.modeling_vit import ViTForImageClassification as StandardViT
from transformer.cds_modules import compute_multiscale_cl_loss
from cds_utils import SupConLoss, TwoCropTransform
from losses import DistributionLoss  # for knowledge distillation
import utils  # distributed training utilities


def get_args_parser():
    """Argument parser for training configuration."""
    parser = argparse.ArgumentParser('BinaryViT + CDS Training', add_help=False)

    # Basic training parameters
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=300, type=int, help='Number of training epochs')

    # Dataset parameters
    parser.add_argument('--data-set', default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'IMNET'],
                        help='Dataset selection')
    parser.add_argument('--data-path', default='./data', type=str, help='Path to dataset')
    parser.add_argument('--input-size', default=224, type=int, help='Input image size')
    parser.add_argument('--cifar-native', action='store_true',
                        help='Use native CIFAR resolution (32x32) without upsampling to input-size')

    # Model parameters
    parser.add_argument('--model', default='./configs/binaryvit-small-patch4-224-cifar10',
                        help='Path to model config directory')
    parser.add_argument('--weight-bits', default=1, type=int, help='Weight quantization bitwidth')
    parser.add_argument('--input-bits', default=1, type=int, help='Activation quantization bitwidth')
    parser.add_argument('--some-fp', action='store_true', help='Keep some layers in full precision')
    parser.add_argument('--avg-res3', action='store_true', help='3x3 average pooling residual at FFN')
    parser.add_argument('--avg-res5', action='store_true', help='5x5 average pooling residual at FFN')
    parser.add_argument('--replace-ln-bn', action='store_true', help='Replace LayerNorm with BatchNorm')
    parser.add_argument('--disable-layerscale', action='store_true', help='Disable LayerScale')
    parser.add_argument('--enable-cls-token', action='store_true', help='Use CLS token instead of avg pooling')

    # CDS parameters
    parser.add_argument('--use-cds', action='store_true', help='Enable CDS auxiliary classifiers')
    parser.add_argument('--use-patch-attention', action='store_true', help='Use PatchAttentionHead classifier')
    parser.add_argument('--use-multiscale-cds', action='store_true', help='Use multi-scale CDS (multiple region tokens per stage)')
    parser.add_argument('--cl', default=0, type=float, help='Contrastive learning loss weight')
    parser.add_argument('--temp', default=0.07, type=float, help='SupConLoss temperature')

    # Teacher model parameters
    parser.add_argument('--teacher-model', default='', type=str, help='Path to teacher model config')
    parser.add_argument('--teacher-model-type', default='', type=str, help='Teacher model type')
    parser.add_argument('--teacher-model-file', default='', type=str, help='Path to teacher model weights')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, help='Optimizer type')
    parser.add_argument('--opt-eps', default=1e-8, type=float, help='Optimizer epsilon')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', help='Optimizer betas')
    parser.add_argument('--clip-grad', type=float, default=None, help='Gradient clipping norm')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, help='LR scheduler type')
    parser.add_argument('--lr', type=float, default=1e-3, help='Base learning rate')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, help='Warmup learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-5, help='Minimum learning rate')
    parser.add_argument('--decay-epochs', type=float, default=30, help='LR decay epoch interval')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--cooldown-epochs', type=int, default=10, help='Number of cooldown epochs')
    parser.add_argument('--patience-epochs', type=int, default=10, help='Patience epochs for Plateau scheduler')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, help='LR decay rate')

    # Data augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, help='Color jitter factor')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', help='AutoAugment policy')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--train-interpolation', type=str, default='bicubic', help='Training interpolation mode')

    # Random Erase parameters
    parser.add_argument('--reprob', type=float, default=0.25, help='Random erase probability')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count')

    # Mixup parameters
    parser.add_argument('--mixup', type=float, default=0.0, help='Mixup alpha (disabled for CDS)')
    parser.add_argument('--cutmix', type=float, default=0.0, help='CutMix alpha (disabled for CDS)')
    parser.add_argument('--mixup-prob', type=float, default=1.0, help='Mixup/cutmix probability')

    # Transfer learning parameters (ImageNet -> CIFAR-10)
    parser.add_argument('--pretrained-model', default='', type=str,
                        help='Path to pretrained model (e.g. ImageNet checkpoint) for transfer learning')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze backbone, train only classification head (for transfer learning)')
    parser.add_argument('--no-strict-load', action='store_true',
                        help='Allow partial weight loading (skip mismatched layers such as classifier head)')

    # System parameters
    parser.add_argument('--output-dir', default='./results/binaryvit_cds_300ep', help='Output directory')
    parser.add_argument('--device', default='cuda', help='Device for training/testing')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--resume', default='', help='Resume from checkpoint path')
    parser.add_argument('--start-epoch', default=0, type=int, help='Start epoch')
    parser.add_argument('--num-workers', default=4, type=int, help='DataLoader worker count')
    parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--world-size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='URL for distributed training setup')

    # Gradient tracking
    parser.add_argument('--track-gradients', action='store_true', help='Track gradient statistics')

    # Miscellaneous
    parser.add_argument('--print-freq', default=50, type=int, help='Print frequency (iterations)')
    parser.add_argument('--model-ema', action='store_true', help='Use Exponential Moving Average')
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='EMA decay rate')

    return parser


def get_cifar_loaders(args):
    """Build CIFAR-10/100 data loaders with appropriate augmentations."""

    # CIFAR normalization statistics
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )

    # Training augmentation
    if args.cifar_native:
        # Native CIFAR 32x32 resolution, no upsampling
        print(f"[Dataset] Using native CIFAR resolution 32x32 (no upsampling)")
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # Upsample to specified resolution
        print(f"[Dataset] Upsampling CIFAR to {args.input_size}x{args.input_size}")
        train_transform = transforms.Compose([
            transforms.Resize(args.input_size),
            transforms.RandomCrop(args.input_size, padding=28),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.Resize(args.input_size),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            normalize,
        ])

    # Wrap with TwoCropTransform for CDS (generates two views per sample)
    if args.use_cds:
        train_transform = TwoCropTransform(train_transform, train_transform)

    # Load dataset
    if args.data_set == 'CIFAR10':
        train_dataset = datasets.CIFAR10(
            root=args.data_path, train=True, download=True, transform=train_transform)
        val_dataset = datasets.CIFAR10(
            root=args.data_path, train=False, download=True, transform=val_transform)
        num_classes = 10
    else:  # CIFAR100
        train_dataset = datasets.CIFAR100(
            root=args.data_path, train=True, download=True, transform=train_transform)
        val_dataset = datasets.CIFAR100(
            root=args.data_path, train=False, download=True, transform=val_transform)
        num_classes = 100

    return train_dataset, val_dataset, num_classes


def get_imagenet_loaders(args):
    """Build ImageNet data loaders using datasets.py build_dataset."""
    from datasets import build_dataset

    dataset_train, num_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    # Apply TwoCropTransform to training set if using CDS
    if args.use_cds:
        original_transform = dataset_train.transform
        dataset_train.transform = TwoCropTransform(original_transform, original_transform)

    return dataset_train, dataset_val, num_classes


def create_model(args, num_classes):
    """Create BinaryViT model from config file."""

    config = ViTConfig.from_pretrained(args.model)

    # Apply model configuration
    config.num_labels = num_classes
    config.weight_bits = args.weight_bits
    config.input_bits = args.input_bits
    config.some_fp = args.some_fp
    config.avg_res3 = args.avg_res3
    config.avg_res5 = args.avg_res5
    config.disable_layerscale = args.disable_layerscale
    config.enable_cls_token = args.enable_cls_token

    # Replace LayerNorm with BatchNorm if specified
    from models import BatchNormT
    config.norm_layer = BatchNormT if args.replace_ln_bn else nn.LayerNorm

    has_depths = hasattr(config, 'depths')
    print(f"[create_model] Model: {args.model}, has_depths: {has_depths}")

    if has_depths:
        # Pyramidal BinaryViT structure
        config.stages = generating_stage_per_depth(config.depths)
        print(f"[create_model] Using PyramidViT with depths: {config.depths}")
        model = PyramidViT(config, use_cds=args.use_cds,
                           use_patch_attention=args.use_patch_attention,
                           use_multiscale_cds=args.use_multiscale_cds)
    else:
        # Standard ViT structure (DeiT) - CDS not supported
        print(f"[create_model] Using StandardViT (CDS not supported for standard ViT)")
        model = StandardViT(config)

    return model


def train_one_epoch(model, teacher_model, criterion, criterion_cl, data_loader, optimizer, device,
                    epoch, loss_scaler, max_norm=None, model_ema=None, args=None, tracker=None):
    """Train the model for one epoch."""

    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    print_freq = args.print_freq if args else 50

    for i, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if args.use_cds:
            # TwoCropTransform returns: ([view1, view2], labels)
            images, labels = data
            images = torch.cat(images, dim=0)  # [2B, 3, H, W]
        else:
            images, labels = data

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        bsz = labels.size(0)

        # Snapshot parameters for gradient tracking
        if tracker is not None and i % print_freq == 0:
            params_before = tracker.get_params_snapshot()

        # Forward pass
        with torch.cuda.amp.autocast():
            if args.use_cds:
                outputs = model(images, return_auxiliary=True)
                logits = outputs.logits
                feat_list = outputs.auxiliary_features if hasattr(outputs, 'auxiliary_features') else None
            else:
                outputs = model(images)
                logits = outputs.logits
                feat_list = None

            # Classification loss (first view only when using CDS)
            logits_main = logits[:bsz] if args.use_cds else logits

            # Knowledge distillation + classification loss
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_images = images[:bsz] if args.use_cds else images

                    # Resize to teacher's expected input size if necessary
                    teacher_unwrapped = teacher_model.module if hasattr(teacher_model, 'module') else teacher_model
                    teacher_expected_size = teacher_unwrapped.config.image_size if hasattr(teacher_unwrapped, 'config') else None

                    if teacher_expected_size is not None and teacher_images.shape[-1] != teacher_expected_size:
                        teacher_images = torch.nn.functional.interpolate(
                            teacher_images, size=(teacher_expected_size, teacher_expected_size),
                            mode='bilinear', align_corners=False
                        )

                    teacher_outputs = teacher_model(teacher_images)

                # Cross-entropy loss with ground truth labels
                loss_ce_label = nn.CrossEntropyLoss()(logits_main, labels)
                # KL-divergence distillation loss
                loss_distill = criterion(logits_main, teacher_outputs.logits)
                # Combined: 80% label CE + 20% distillation
                loss_ce = 0.8 * loss_ce_label + 0.2 * loss_distill
            else:
                loss_ce = criterion(logits_main, labels)
                loss_ce_label = loss_ce.item()
                loss_distill = 0.0

            # Contrastive learning loss (CDS)
            loss_cl = 0
            if args.use_cds and feat_list is not None and len(feat_list) > 0:
                if args.use_multiscale_cds and isinstance(feat_list[0], dict):
                    # Multi-scale CDS: use dedicated loss computation function
                    loss_cl = compute_multiscale_cl_loss(feat_list, labels, criterion_cl, bsz)
                else:
                    # Standard CDS: one global feature per stage
                    for stage_idx, features in enumerate(feat_list):
                        # features: [2B, 512] where B=bsz (doubled by TwoCropTransform)
                        expected_size = 2 * bsz

                        if features.shape[0] != expected_size or features.shape[1] != 512:
                            if i == 0:  # Print warning only on first iteration
                                print(f"[Warning] Stage {stage_idx} feature shape: {features.shape}, expected: [{expected_size}, 512]")
                            if len(features.shape) != 2:
                                print(f"[Error] Stage {stage_idx} feature has wrong dimensions: {features.shape}")
                                continue

                        # Split into two views: [2B, 512] -> [B, 512], [B, 512]
                        try:
                            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                            # Stack into [B, 2, 512]
                            features_pair = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                            stage_cl_loss = criterion_cl(features_pair, labels)
                            loss_cl += stage_cl_loss
                        except RuntimeError as e:
                            print(f"[Error] Stage {stage_idx} split failed: {e}")
                            continue

                if loss_cl != 0:
                    loss_cl *= args.cl

            # Total loss
            loss = loss_ce + loss_cl

        loss_value = loss.item()

        if not torch.isfinite(torch.tensor(loss_value)):
            print(f"Loss is {loss_value}, stopping training")
            raise ValueError(f"Loss is {loss_value}, stopping training")

        # Backward pass
        optimizer.zero_grad()

        # Mixed precision backward with NativeScaler
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        # Update EMA
        if model_ema is not None:
            model_ema.update(model)

        # Record gradients
        if tracker is not None and i % print_freq == 0:
            tracker.record_gradients(epoch)
            tracker.record_updates(epoch, params_before)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_ce=loss_ce.item())

        if teacher_model is not None:
            metric_logger.update(loss_distill=loss_distill.item())
            metric_logger.update(loss_ce_label=loss_ce_label.item())

        if args.use_cds and isinstance(loss_cl, torch.Tensor):
            metric_logger.update(loss_cl=loss_cl.item())

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # Synchronize metrics across processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args=None):
    """Evaluate model on validation set."""

    criterion = nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    # Get model's expected input resolution
    model_unwrapped = model.module if hasattr(model, 'module') else model
    expected_size = model_unwrapped.config.image_size if hasattr(model_unwrapped, 'config') else None

    for images, labels in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Upsample if model expects different resolution (e.g. teacher at 224 but data is 32)
        if expected_size is not None and images.shape[-1] != expected_size:
            images = torch.nn.functional.interpolate(
                images, size=(expected_size, expected_size),
                mode='bilinear', align_corners=False
            )

        with torch.cuda.amp.autocast():
            # PyramidViT supports return_auxiliary, StandardViT does not
            if hasattr(model_unwrapped, 'use_cds'):
                outputs = model(images, return_auxiliary=False)
            else:
                outputs = model(images)

            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            loss = criterion(logits, labels)

        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print(f'* Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} loss {metric_logger.loss.global_avg:.3f}')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def main(args):
    """Main training function."""

    # Initialize distributed training
    utils.init_distributed_mode(args)

    print(args)

    # Save training config to log.txt (main process only)
    if utils.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        log_file = os.path.join(args.output_dir, 'log.txt')

        with open(log_file, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Training started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
            f.write("Training arguments:\n")
            f.write("-" * 80 + "\n")
            for key, value in sorted(vars(args).items()):
                f.write(f"  {key}: {value}\n")
            f.write("=" * 80 + "\n\n")

        print(f"Training args saved to: {log_file}")

    device = torch.device(args.device)

    # Set random seeds for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Load dataset
    if args.data_set == 'IMNET':
        dataset_train, dataset_val, args.nb_classes = get_imagenet_loaders(args)
    else:  # CIFAR10 or CIFAR100
        dataset_train, dataset_val, args.nb_classes = get_cifar_loaders(args)

    # Distributed samplers
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )

    # Data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # Create model
    print(f"Creating model: {args.model}")
    model = create_model(args, args.nb_classes)
    print(model)
    model.to(device)

    # Sync BatchNorm across GPUs
    if args.replace_ln_bn:
        from models import SyncBatchNormT
        model = SyncBatchNormT.convert_sync_batchnorm(model)

    # Exponential Moving Average
    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(model, decay=args.model_ema_decay, device='', resume='')

    # Wrap with DDP
    model_without_ddp = model
    if args.distributed:
        # find_unused_parameters=True is needed for CDS since some auxiliary
        # classifier parameters may be skipped in error handling paths
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {n_parameters}')

    # Freeze backbone for transfer learning
    if args.freeze_backbone:
        print("Freezing backbone (training classification head only)")
        frozen_params = 0
        trainable_params = 0

        for name, param in model_without_ddp.named_parameters():
            # Freeze all parameters except classification head
            if not any(keyword in name for keyword in ['classifier', 'head', 'fc']):
                param.requires_grad = False
                frozen_params += param.numel()
            else:
                param.requires_grad = True
                trainable_params += param.numel()
                print(f"  Trainable: {name} ({param.numel()} params)")

        print(f"  Frozen: {frozen_params:,} params, Trainable: {trainable_params:,} params")
        print(f"  Trainable ratio: {trainable_params / (frozen_params + trainable_params) * 100:.2f}%")
        n_parameters = trainable_params

    # Linear LR scaling
    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr

    # Optimizer, scaler, and scheduler
    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    # Loss functions
    if args.teacher_model:
        criterion = DistributionLoss()  # KL divergence for knowledge distillation
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    criterion_cl = SupConLoss(temperature=args.temp).to(device) if args.use_cds else None

    # Gradient tracker (optional)
    tracker = None
    if args.track_gradients and utils.is_main_process():
        try:
            from cds_utils import GradientTracker
            gradient_log_dir = os.path.join(args.output_dir, 'gradient_logs')
            tracker = GradientTracker(model_without_ddp, save_dir=gradient_log_dir)
            print(f"Gradient tracking enabled: {gradient_log_dir}")
        except (ImportError, AttributeError):
            print("GradientTracker not available, skipping gradient tracking.")

    output_dir = Path(args.output_dir)
    max_accuracy = 0.0

    # Load pretrained weights for transfer learning (e.g. ImageNet -> CIFAR-10)
    if args.pretrained_model:
        print(f"Loading pretrained model from: {args.pretrained_model}")
        checkpoint = torch.load(args.pretrained_model, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

        if args.no_strict_load:
            model_state_dict = model_without_ddp.state_dict()
            filtered_state_dict = {}
            skipped_keys = []
            for k, v in state_dict.items():
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        filtered_state_dict[k] = v
                    else:
                        skipped_keys.append(f"{k} (shape mismatch: {v.shape} vs {model_state_dict[k].shape})")
                else:
                    skipped_keys.append(f"{k} (not in model)")

            print(f"Loaded {len(filtered_state_dict)} layers from pretrained model")
            if skipped_keys:
                print(f"Skipped {len(skipped_keys)} layers:")
                for key in skipped_keys[:5]:
                    print(f"   - {key}")
                if len(skipped_keys) > 5:
                    print(f"   ... and {len(skipped_keys) - 5} more")
            model_without_ddp.load_state_dict(filtered_state_dict, strict=False)
        else:
            model_without_ddp.load_state_dict(state_dict)

        test_stats = evaluate(data_loader_val, model, device, args)
        print(f"Pretrained model accuracy: {test_stats['acc1']:.1f}%")
        max_accuracy = test_stats['acc1']

    # Resume from checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.step(args.start_epoch - 1)

        test_stats = evaluate(data_loader_val, model, device, args)
        print(f"Accuracy of checkpoint: {test_stats['acc1']:.1f}%")
        max_accuracy = test_stats['acc1']

    # Load teacher model for knowledge distillation
    teacher_model = None
    if args.teacher_model:
        print(f"Loading teacher model: {args.teacher_model}")

        # Temporarily modify args to create teacher model
        original_model = args.model
        original_weight_bits = args.weight_bits
        original_input_bits = args.input_bits

        args.model = args.teacher_model
        args.weight_bits = 32  # Teacher uses full precision
        args.input_bits = 32

        teacher_model = create_model(args, args.nb_classes)

        # Restore original args
        args.model = original_model
        args.weight_bits = original_weight_bits
        args.input_bits = original_input_bits

        teacher_model.to(device)
        teacher_model.eval()
        teacher_model_without_ddp = teacher_model
        if args.distributed:
            teacher_model = torch.nn.parallel.DistributedDataParallel(
                teacher_model, device_ids=[args.gpu], find_unused_parameters=True
            )
            teacher_model_without_ddp = teacher_model.module

        best_teacher_model = torch.load(args.teacher_model_file, map_location='cpu')
        if 'model' in best_teacher_model:
            best_teacher_model = best_teacher_model['model']
        teacher_model_without_ddp.load_state_dict(best_teacher_model)

        test_stats = evaluate(data_loader_val, teacher_model, device, args)
        print(f"Teacher model accuracy: {test_stats['acc1']:.1f}%")

    # Training loop
    print("Starting training")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, teacher_model, criterion, criterion_cl, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, args, tracker
        )

        # Record losses for gradient tracker
        if tracker is not None:
            tracker.record_loss(epoch, 'train_loss', train_stats['loss'])
            tracker.record_loss(epoch, 'train_ce', train_stats['loss_ce'])
            if args.use_cds:
                tracker.record_loss(epoch, 'train_cl', train_stats.get('loss_cl', 0))

        lr_scheduler.step(epoch)

        # Save checkpoint
        if args.output_dir:
            checkpoint_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            if args.model_ema:
                checkpoint_dict['model_ema'] = get_state_dict(model_ema)
            utils.save_on_master(checkpoint_dict, output_dir / 'checkpoint.pth')

        # Evaluate
        test_stats = evaluate(data_loader_val, model, device, args)
        print(f"Accuracy: {test_stats['acc1']:.1f}%")

        if tracker is not None:
            tracker.record_loss(epoch, 'val_loss', test_stats['loss'])
            tracker.record_loss(epoch, 'val_acc1', test_stats['acc1'])

        # Save best model
        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            if args.output_dir:
                best_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }
                if args.model_ema:
                    best_dict['model_ema'] = get_state_dict(model_ema)
                utils.save_on_master(best_dict, output_dir / 'best.pth')

        print(f'Max accuracy: {max_accuracy:.2f}%')

        # Write log
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    # Save gradient tracking data
    if tracker is not None:
        exp_name = os.path.basename(args.output_dir)
        tracker.save(f'gradient_tracker_{exp_name}.pkl')
        tracker.save_summary(f'gradient_summary_{exp_name}.json')
        print("Gradient tracking data saved.")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time: {total_time_str}')
    print(f'Max accuracy: {max_accuracy:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BinaryViT + CDS Training', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

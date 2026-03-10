#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supervised Contrastive Loss and TwoCropTransform for BinaryViT CDS

This module provides two key components for Contrastive Deep Supervision:
1. SupConLoss: Supervised contrastive loss function
2. TwoCropTransform: Data augmentation that generates two views of each image

Reference: Supervised Contrastive Learning (https://arxiv.org/pdf/2004.11362.pdf)

Usage in CDS:
    # Create loss function
    criterion_cl = SupConLoss(temperature=0.07)

    # In training loop, for each auxiliary classifier output
    for aux_features in auxiliary_outputs:
        # aux_features: [B, n_views, 512] normalized features
        cl_loss = criterion_cl(aux_features, labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.

    This loss pulls together embeddings from the same class and pushes apart
    embeddings from different classes. It also supports unsupervised contrastive
    loss (SimCLR) when labels are not provided.

    Args:
        temperature: Temperature parameter for scaling similarity (default: 0.07)
        contrast_mode: 'one' or 'all' - whether to use one or all views as anchor
        base_temperature: Base temperature for loss scaling (default: 0.07)

    Forward Args:
        features: [bsz, n_views, feat_dim] - normalized feature vectors
        labels: [bsz] - class labels for supervised learning
        mask: [bsz, bsz] - optional custom contrastive mask

    Returns:
        Scalar contrastive loss
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  #   2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


class TwoCropTransform:
    """Two-crop data augmentation transform for contrastive learning

    Generates two augmented views of the same image for contrastive learning.
    Each view undergoes different random augmentations (crop, flip, color jitter, etc.)

    Args:
        transform1: First augmentation pipeline (torchvision.transforms)
        transform2: Second augmentation pipeline (can be same as transform1)

    Returns:
        List of two augmented views: [view1, view2]

    Example:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        two_crop = TwoCropTransform(train_transform, train_transform)
        view1, view2 = two_crop(image)
    """
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        return [self.transform1(x), self.transform2(x)]


# === Testing utilities ===

def test_supcon_loss():
    """Test SupConLoss with mock features"""
    print("\n=== Testing SupConLoss ===")

    # Mock normalized features: [batch_size, n_views, feat_dim]
    batch_size = 8
    n_views = 2
    feat_dim = 512

    features = torch.randn(batch_size, n_views, feat_dim)
    features = F.normalize(features, dim=2)  # Normalize features

    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])  # 4 classes, 2 samples each

    criterion = SupConLoss(temperature=0.07)
    loss = criterion(features, labels)

    print(f"Batch size: {batch_size}, Views: {n_views}, Feature dim: {feat_dim}")
    print(f"Labels: {labels.tolist()}")
    print(f"SupConLoss: {loss.item():.4f}")

    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"

    print("✅ SupConLoss test passed!")


def test_two_crop_transform():
    """Test TwoCropTransform"""
    print("\n=== Testing TwoCropTransform ===")

    from torchvision import transforms
    from PIL import Image
    import numpy as np

    # Create a simple transform
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    two_crop = TwoCropTransform(transform, transform)

    # Create a mock PIL image
    img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    view1, view2 = two_crop(img)

    print(f"Input image size: {img.size}")
    print(f"View 1 shape: {view1.shape}")
    print(f"View 2 shape: {view2.shape}")

    assert view1.shape == torch.Size([3, 32, 32]), "View 1 shape mismatch"
    assert view2.shape == torch.Size([3, 32, 32]), "View 2 shape mismatch"
    assert not torch.equal(view1, view2), "Two views should be different (random augmentation)"

    print("✅ TwoCropTransform test passed!")


if __name__ == "__main__":
    print("Testing Supervised Contrastive Loss and TwoCropTransform")
    print("=" * 60)

    test_supcon_loss()
    test_two_crop_transform()

    print("\n" + "=" * 60)
    print("✅ All tests passed!")

"""
Contrastive Deep Supervision (CDS) Modules for BinaryViT

This module implements auxiliary classifiers for BinaryViT, adapted from
the successful CDS design in BNN ResNet (BNN_CDS/cifar/models_cifar/resnet2.py).

Key components:
- SepConv: Standard depthwise separable convolution (proven best in ResNet experiments)
- ViTSepConv: Adapter to convert ViT sequence features to spatial format for SepConv
- ViTAuxiliaryClassifier: Complete auxiliary classifier network for 4-stage pyramidal ViT

Design decisions based on experimental validation:
- Use standard ReLU (not RPReLU) - simpler and more stable
- Gradual channel expansion (64→64→128, not 64→128→128) - more effective
- No residual connections in auxiliary classifiers - stronger supervision signal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SepConv(nn.Module):
    """Standard Depthwise Separable Convolution for auxiliary classifiers

    This is the standard design that achieved 93.98% accuracy on CIFAR-10 in ResNet experiments.
    Architecture: Two depthwise-pointwise blocks with gradual channel expansion.

    Args:
        channel_in: Input channels
        channel_out: Output channels
        kernel_size: Convolution kernel size (default: 3)
        stride: Stride for first depthwise conv (default: 2 for downsampling)
        padding: Padding for depthwise conv (default: 1)
        affine: Whether to use affine in BatchNorm (default: True)
    """
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()

        self.op = nn.Sequential(
            # First depthwise-pointwise block with downsampling
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),

            # Second depthwise-pointwise block
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size,
                      stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] spatial features
        Returns:
            out: [B, C_out, H', W'] downsampled features
        """
        return self.op(x)


class ViTSepConv(nn.Module):
    """Adapter to apply SepConv to ViT sequence features

    Transforms ViT's sequence format [B, N, D] to spatial format [B, D, H, W],
    applies SepConv, and returns spatial features [B, D_out, H', W'] for further processing.

    Args:
        dim_in: Input dimension (ViT hidden size)
        dim_out: Output dimension
        num_patches: Number of patches (N = H * W)
        stride: Stride for SepConv downsampling (default: 2)
    """
    def __init__(self, dim_in, dim_out, num_patches, stride=2):
        super(ViTSepConv, self).__init__()
        self.num_patches = num_patches

        # Calculate spatial dimensions (assuming square patch grid)
        self.H = self.W = int(math.sqrt(num_patches))

        if self.H * self.W != num_patches:
            raise ValueError(f"num_patches ({num_patches}) must be a perfect square for spatial reshape")

        # Standard SepConv (reuse proven design from ResNet)
        self.sepconv = SepConv(
            channel_in=dim_in,
            channel_out=dim_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            affine=True
        )

    def forward(self, x):
        """
        Args:
            x: [B, N, D] - ViT sequence format
        Returns:
            features: [B, D_out, H', W'] - Spatial features after SepConv
        """
        B, N, D = x.shape

        # Reshape: [B, N, D] → [B, D, H, W]
        # N = H * W, so we transpose and reshape
        x = x.transpose(1, 2).reshape(B, D, self.H, self.W)

        # Apply SepConv: [B, D, H, W] → [B, D_out, H', W']
        x = self.sepconv(x)

        return x


class ViTAuxiliaryClassifier(nn.Module):
    """Complete auxiliary classifier network for pyramidal ViT with CDS

    Creates auxiliary classifiers for each stage of BinaryViT's pyramidal architecture.
    Each auxiliary classifier progressively downsamples features to a common dimension (512)
    for contrastive learning.

    Architecture (for 224x224 input):
        Stage 0: [B, 3136, 64]  (56x56) → 512  (3 SepConv: 64→128→256→512)
        Stage 1: [B, 784, 128]  (28x28) → 512  (2 SepConv: 128→256→512)
        Stage 2: [B, 196, 256]  (14x14) → 512  (1 SepConv: 256→512)
        Stage 3: [B, 49, 512]   (7x7)   → 512  (Direct pool)

    For CIFAR (32x32 input), patch sizes will be different but structure is the same.

    Args:
        config: ViT configuration with:
            - hidden_size: List of dimensions [64, 128, 256, 512]
            - image_size: Input image size (224 for ImageNet, 32 for CIFAR)
            - patch_size: Patch size (default: 4)
    """
    def __init__(self, config):
        super(ViTAuxiliaryClassifier, self).__init__()
        self._debug_printed = False  # Flag to print debug info only once

        self.hidden_size = config.hidden_size
        self.image_size = config.image_size if hasattr(config, 'image_size') else 224
        self.patch_size = config.patch_size if hasattr(config, 'patch_size') else 4

        # Calculate patch grid sizes at each stage
        # Initial patches after patch embedding
        init_patch_size = self.image_size // self.patch_size

        # After each stage, spatial size is halved by BinaryPatchEmbed (2x2, stride=2)
        # Dynamically support 3-stage and 4-stage models
        num_stages = len(self.hidden_size)
        self.patch_sizes = []
        for i in range(num_stages):
            stage_patch_size = (init_patch_size // (2 ** i)) ** 2
            self.patch_sizes.append(stage_patch_size)
        # Examples:
        # 3-stage CIFAR (32, patch=2): [16x16=256, 8x8=64, 4x4=16]
        # 4-stage ImageNet (224, patch=4): [56x56=3136, 28x28=784, 14x14=196, 7x7=49]

        # Target dimension for all auxiliary features (for contrastive learning)
        target_dim = 512

        # Dynamically build auxiliary classifiers, supporting both 3-stage and 4-stage models
        self.auxiliaries = nn.ModuleList()

        for stage_idx in range(num_stages):
            in_dim = self.hidden_size[stage_idx]
            patch_size_tokens = self.patch_sizes[stage_idx]

            # Compute number of downsampling steps needed to reach target spatial size (7x7 or 4x4)
            # 3-stage model: Stage 0->4x4 needs 2, Stage 1->4x4 needs 1, Stage 2 needs none
            # 4-stage model: Stage 0->7x7 needs 3, Stage 1->7x7 needs 2, Stage 2->7x7 needs 1, Stage 3 needs none
            spatial_size = int(patch_size_tokens ** 0.5)
            target_spatial = 4 if num_stages == 3 else 7
            num_downsamples = 0
            temp_size = spatial_size
            while temp_size > target_spatial:
                temp_size //= 2
                num_downsamples += 1

            # Build downsampling path
            if num_downsamples == 0:
                # No downsampling needed: use a Linear projection
                # [B, N, D] -> [B, N, 512] -> [B, 512]
                proj_layer = nn.Sequential(
                    nn.Linear(in_dim, target_dim),
                    nn.LayerNorm(target_dim)
                )
                self.auxiliaries.append(proj_layer)
            else:
                # Downsampling needed: use Conv path
                layers = []
                current_dim = in_dim

                for ds_idx in range(num_downsamples):
                    # Progressively double channels up to target_dim
                    if ds_idx == num_downsamples - 1:
                        # Final layer reaches target_dim
                        out_dim = target_dim
                    else:
                        # Intermediate layers double channels
                        out_dim = min(current_dim * 2, target_dim)

                    if ds_idx == 0:
                        # First layer uses ViTSepConv (sequence -> spatial)
                        layers.append(ViTSepConv(current_dim, out_dim, patch_size_tokens, stride=2))
                    else:
                        # Subsequent layers use standard SepConv
                        layers.append(SepConv(current_dim, out_dim, stride=2))

                    current_dim = out_dim

                # Add 1x1 conv to adjust final dim to target_dim if needed
                if current_dim != target_dim:
                    layers.append(nn.Conv2d(current_dim, target_dim, kernel_size=1))

                self.auxiliaries.append(nn.Sequential(*layers))

        print(f"[CDS] Initialized ViTAuxiliaryClassifier:")
        print(f"  - Image size: {self.image_size}")
        print(f"  - Num stages: {num_stages}")
        print(f"  - Patch sizes: {self.patch_sizes}")
        print(f"  - Hidden sizes: {self.hidden_size}")
        print(f"  - Target dim: {target_dim}")

    def forward(self, stage_features):
        """
        Args:
            stage_features: List of N tensors, one from each stage (N=3 or 4)
                - stage_features[i]: [B, Ni, Di] from stage i

        Returns:
            feat_list: List of N tensors [B, 512] normalized features for contrastive learning
        """
        feat_list = []

        if not self._debug_printed:
            print(f"[DEBUG CDS] Processing {len(stage_features)} stage features:")
            for idx, feat in enumerate(stage_features):
                print(f"  Input Stage {idx}: shape = {feat.shape}")

        # Process all stages dynamically
        for stage_idx, features in enumerate(stage_features):
            # Apply auxiliary classifier if available for this stage
            if stage_idx < len(self.auxiliaries):
                auxiliary = self.auxiliaries[stage_idx]

                # Determine processing path based on first layer type
                first_layer = auxiliary[0] if isinstance(auxiliary, nn.Sequential) and len(auxiliary) > 0 else auxiliary

                if isinstance(first_layer, nn.Linear):
                    # Linear projection path: operate directly on sequence format
                    # [B, N, D] -> [B, N, 512]
                    if not self._debug_printed:
                        print(f"[DEBUG CDS] Stage {stage_idx} using Linear path")
                    features = auxiliary(features)
                    if not self._debug_printed:
                        print(f"[DEBUG CDS] Stage {stage_idx} after Linear: {features.shape}")
                    # Average pooling: [B, N, 512] -> [B, 512]
                    features = features.mean(dim=1)
                    if not self._debug_printed:
                        print(f"[DEBUG CDS] Stage {stage_idx} after mean: {features.shape}")
                else:
                    # Conv path: convert to spatial format first, then apply layers
                    if not self._debug_printed:
                        print(f"[DEBUG CDS] Stage {stage_idx} using Conv path")
                    for layer in auxiliary:
                        features = layer(features)
                        if not self._debug_printed:
                            print(f"[DEBUG CDS] Stage {stage_idx} after layer {type(layer)}: {features.shape}")
                    # Spatial pooling: [B, C, H, W] -> [B, C]
                    features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
                    if not self._debug_printed:
                        print(f"[DEBUG CDS] Stage {stage_idx} after pooling: {features.shape}")

                # L2 normalization for contrastive learning
                features = F.normalize(features, dim=1)
                if not self._debug_printed:
                    print(f"[DEBUG CDS] Stage {stage_idx} final output: {features.shape}")
                feat_list.append(features)

        self._debug_printed = True  # Set flag after first forward pass

        return feat_list


class MultiScaleViTAuxiliaryClassifier(nn.Module):
    """Multi-scale region token CDS auxiliary classifier

    Extracts multiple regional features per stage (instead of a single global feature),
    providing richer supervision signals for contrastive learning.

    Strategy:
        - Stage 0-2 (high resolution): 5 tokens (1 global + 4 regions)
        - Stage 3 (7x7, too small): 1 token (global only)

    Args:
        config: ViT configuration
        num_regions: Number of regions (default 4, i.e. 2x2 grid)
    """
    def __init__(self, config, num_regions=4):
        super(MultiScaleViTAuxiliaryClassifier, self).__init__()
        self._debug_printed = False

        self.hidden_size = config.hidden_size
        self.image_size = config.image_size if hasattr(config, 'image_size') else 224
        self.patch_size = config.patch_size if hasattr(config, 'patch_size') else 4
        self.num_regions = num_regions
        self.grid_size = int(math.sqrt(num_regions))  # 2 for 4 regions

        # Calculate number of patches per stage
        init_patch_size = self.image_size // self.patch_size
        num_stages = len(self.hidden_size)
        self.patch_sizes = []
        for i in range(num_stages):
            stage_patch_size = (init_patch_size // (2 ** i)) ** 2
            self.patch_sizes.append(stage_patch_size)

        # Target dimension
        target_dim = 512
        self.target_dim = target_dim

        # Create a projection layer per stage (project different hidden sizes to unified 512)
        self.projectors = nn.ModuleList()
        for stage_idx in range(num_stages):
            in_dim = self.hidden_size[stage_idx]
            self.projectors.append(nn.Sequential(
                nn.Linear(in_dim, target_dim),
                nn.LayerNorm(target_dim)
            ))

        # Decide which stages use multi-scale extraction
        # Stage 3 (7x7) is too small for region partitioning
        self.use_multiscale = []
        for stage_idx in range(num_stages):
            spatial_size = int(self.patch_sizes[stage_idx] ** 0.5)
            # Only use multi-scale when spatial size >= 8
            self.use_multiscale.append(spatial_size >= 8)

        print(f"[Multi-Scale CDS] Initialized:")
        print(f"  - Num stages: {num_stages}")
        print(f"  - Patch sizes: {self.patch_sizes}")
        print(f"  - Use multiscale: {self.use_multiscale}")
        print(f"  - Num regions: {num_regions}")

    def _extract_region_features(self, features, stage_idx):
        """Extract regional features from a stage's output.

        Args:
            features: [B, N, D] - stage features
            stage_idx: stage index

        Returns:
            list of [B, D] tensors - 1 global + num_regions regional features
        """
        B, N, D = features.shape
        H = W = int(N ** 0.5)

        result = []

        # 1. Global feature (mean pooling over all tokens)
        global_feat = features.mean(dim=1)  # [B, D]
        result.append(global_feat)

        # 2. Regional features (only when multi-scale is enabled for this stage)
        if self.use_multiscale[stage_idx]:
            spatial = features.reshape(B, H, W, D)
            h_step = H // self.grid_size
            w_step = W // self.grid_size

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    region = spatial[:, i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step, :]
                    region_feat = region.reshape(B, -1, D).mean(dim=1)  # [B, D]
                    result.append(region_feat)

        return result

    def forward(self, stage_features):
        """
        Args:
            stage_features: List of N tensors from each stage
                - stage_features[i]: [B, Ni, Di]

        Returns:
            feat_list: List of dicts, one per stage
                - 'global': [B, 512] normalized global feature
                - 'regions': List of [B, 512] normalized region features (if multiscale)
                - 'all': List of all [B, 512] features for easy iteration
        """
        feat_list = []

        if not self._debug_printed:
            print(f"[DEBUG Multi-Scale CDS] Processing {len(stage_features)} stages")

        for stage_idx, features in enumerate(stage_features):
            if stage_idx >= len(self.projectors):
                continue

            # Extract multi-scale features
            raw_features = self._extract_region_features(features, stage_idx)

            # Project and normalize
            projector = self.projectors[stage_idx]
            projected = []
            for feat in raw_features:
                proj = projector(feat)  # [B, 512]
                proj = F.normalize(proj, dim=1)  # L2 normalize
                projected.append(proj)

            # Organize outputs
            stage_output = {
                'global': projected[0],
                'regions': projected[1:] if len(projected) > 1 else [],
                'all': projected  # list of all features for easy iteration
            }

            feat_list.append(stage_output)

        self._debug_printed = True
        return feat_list


def compute_multiscale_cl_loss(feat_list, labels, criterion_cl, bsz):
    """Compute multi-scale contrastive learning loss.

    Args:
        feat_list: Output from MultiScaleViTAuxiliaryClassifier
        labels: [B] class labels
        criterion_cl: SupConLoss instance
        bsz: Original batch size (before TwoCropTransform doubling)

    Returns:
        total_loss: Scalar loss (averaged over all feature tokens across all stages)
    """
    total_loss = 0
    num_losses = 0

    for stage_idx, stage_output in enumerate(feat_list):
        all_features = stage_output['all']

        for feat_idx, features in enumerate(all_features):
            # features: [2B, 512] (two views concatenated by TwoCropTransform)
            # Split into two views
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            # Reorganize as [B, 2, 512]
            features_pair = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            # Compute contrastive loss
            loss = criterion_cl(features_pair, labels)
            total_loss += loss
            num_losses += 1

    if num_losses > 0:
        total_loss = total_loss / num_losses

    return total_loss


# === Testing utilities ===

def test_sepconv():
    """Test SepConv module"""
    print("\n=== Testing SepConv ===")

    sepconv = SepConv(channel_in=64, channel_out=128, stride=2)
    x = torch.randn(2, 64, 56, 56)

    out = sepconv(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected: [2, 128, 28, 28]")

    assert out.shape == (2, 128, 28, 28), "SepConv output shape mismatch"
    print("✅ SepConv test passed!")


def test_vit_sepconv():
    """Test ViTSepConv module"""
    print("\n=== Testing ViTSepConv ===")

    vit_sepconv = ViTSepConv(dim_in=64, dim_out=128, num_patches=3136, stride=2)
    x = torch.randn(2, 3136, 64)  # [B, N, D]

    out = vit_sepconv(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected: [2, 128, 28, 28]")

    assert out.shape == (2, 128, 28, 28), "ViTSepConv output shape mismatch"
    print("✅ ViTSepConv test passed!")


def test_auxiliary_classifier():
    """Test ViTAuxiliaryClassifier module"""
    print("\n=== Testing ViTAuxiliaryClassifier ===")

    # Create mock config
    class Config:
        hidden_size = [64, 128, 256, 512]
        image_size = 224
        patch_size = 4

    config = Config()
    aux_classifier = ViTAuxiliaryClassifier(config)

    # Create mock stage features
    stage_features = [
        torch.randn(2, 3136, 64),   # Stage 0: 56x56
        torch.randn(2, 784, 128),   # Stage 1: 28x28
        torch.randn(2, 196, 256),   # Stage 2: 14x14
        torch.randn(2, 49, 512),    # Stage 3: 7x7
    ]

    feat_list = aux_classifier(stage_features)

    print(f"\nStage features:")
    for i, feat in enumerate(stage_features):
        print(f"  Stage {i}: {feat.shape}")

    print(f"\nAuxiliary outputs:")
    for i, feat in enumerate(feat_list):
        print(f"  Auxiliary {i}: {feat.shape}")
        # Check normalization
        norm = torch.norm(feat, dim=1)
        print(f"    L2 norm: mean={norm.mean():.4f}, std={norm.std():.4f} (should be ~1.0)")

    # Verify shapes
    assert len(feat_list) == 4, "Should have 4 auxiliary features"
    for i, feat in enumerate(feat_list):
        assert feat.shape == (2, 512), f"Auxiliary {i} shape mismatch: {feat.shape}"

    print("\n✅ ViTAuxiliaryClassifier test passed!")


if __name__ == "__main__":
    print("Testing CDS Modules for BinaryViT")
    print("=" * 50)

    test_sepconv()
    test_vit_sepconv()
    test_auxiliary_classifier()

    print("\n" + "=" * 50)
    print("✅ All tests passed!")

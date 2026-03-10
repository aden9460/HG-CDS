
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RegionTokenExtractor(nn.Module):


    def __init__(self, num_regions=4, use_fine_grain=False):

        super().__init__()
        self.num_regions = num_regions
        self.use_fine_grain = use_fine_grain
        self.grid_size = int(math.sqrt(num_regions))

    def forward(self, sequence_features):

        B, N, D = sequence_features.shape
        H = W = int(math.sqrt(N))


        spatial = sequence_features.reshape(B, H, W, D)


        global_feat = sequence_features.mean(dim=1)  # [B, D]

        regions = []
        h_step = H // self.grid_size
        w_step = W // self.grid_size

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                region = spatial[:, i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step, :]
                region_feat = region.reshape(B, -1, D).mean(dim=1)  # [B, D]
                regions.append(region_feat)


        fine_features = None
        if self.use_fine_grain and H >= 8:  
            fine_features = []
            fine_grid = 4  
            h_fine = H // fine_grid
            w_fine = W // fine_grid

            for i in range(fine_grid):
                for j in range(fine_grid):
                    fine = spatial[:, i*h_fine:(i+1)*h_fine, j*w_fine:(j+1)*w_fine, :]
                    fine_feat = fine.reshape(B, -1, D).mean(dim=1)
                    fine_features.append(fine_feat)

        return {
            'global': global_feat,
            'regions': regions,
            'fine': fine_features
        }


class MultiScaleClassificationHead(nn.Module):


    def __init__(self, hidden_size, num_classes, num_regions=4, fusion_method='attention'):

        super().__init__()
        self.num_regions = num_regions
        self.fusion_method = fusion_method

  
        self.region_extractor = RegionTokenExtractor(num_regions=num_regions)

        if fusion_method == 'attention':

            self.attention_fusion = nn.MultiheadAttention(
                hidden_size, num_heads=8, batch_first=True
            )
            self.classifier = nn.Linear(hidden_size, num_classes)

        elif fusion_method == 'concat':

            self.classifier = nn.Linear(hidden_size * (num_regions + 1), num_classes)

        else:  
            self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, sequence_output):
  
        ms_features = self.region_extractor(sequence_output)

        if self.fusion_method == 'attention':

            all_features = [ms_features['global']] + ms_features['regions']
            features = torch.stack(all_features, dim=1)  # [B, num_regions+1, D]


            fused, _ = self.attention_fusion(features, features, features)

            logits = self.classifier(fused[:, 0, :])

        elif self.fusion_method == 'concat':
            # 拼接所有特征
            all_features = [ms_features['global']] + ms_features['regions']
            concat_features = torch.cat(all_features, dim=-1)  # [B, D*(num_regions+1)]
            logits = self.classifier(concat_features)

        else:  # 'mean'
            # 平均所有特征
            all_features = [ms_features['global']] + ms_features['regions']
            mean_features = torch.stack(all_features, dim=1).mean(dim=1)  # [B, D]
            logits = self.classifier(mean_features)

        return logits


class MultiScaleCDSAuxiliary(nn.Module):
    """
    CDS辅助分类器的多尺度增强版本
    用于生成多个特征供对比学习
    """

    def __init__(self, input_dim, output_dim=512, num_regions=4, stage_idx=0):
        """
        Args:
            input_dim: 输入维度
            output_dim: 输出维度（用于对比学习）
            num_regions: 区域数量
            stage_idx: Stage索引（决定使用的粒度）
        """
        super().__init__()
        self.stage_idx = stage_idx

        # 根据stage决定粒度
        use_fine = (stage_idx == 0)  # 只在第一个stage使用细粒度
        self.extractor = RegionTokenExtractor(
            num_regions=num_regions,
            use_fine_grain=use_fine
        )

        # 投影层：将不同维度投影到统一的output_dim
        self.projector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, features):
        """
        Args:
            features: [B, N, D] Stage特征

        Returns:
            dict: 多尺度特征，每个都是[B, output_dim]并且L2归一化
        """
        # 提取多尺度特征
        ms_features = self.extractor(features)

        # 投影并归一化
        output = {}

        # 全局特征
        global_proj = self.projector(ms_features['global'])
        output['global'] = F.normalize(global_proj, dim=1)

        # 区域特征
        region_projs = []
        for region_feat in ms_features['regions']:
            proj = self.projector(region_feat)
            region_projs.append(F.normalize(proj, dim=1))
        output['regions'] = region_projs

        # 细粒度特征（如果有）
        if ms_features['fine'] is not None:
            fine_projs = []
            for fine_feat in ms_features['fine']:
                proj = self.projector(fine_feat)
                fine_projs.append(F.normalize(proj, dim=1))
            output['fine'] = fine_projs

        return output


class MultiScaleSupConLoss(nn.Module):
    """
    多尺度监督对比损失
    支持不同数量的tokens进行对比学习
    """

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features_dict, labels, weights=None):
        """
        Args:
            features_dict: 字典，包含'global', 'regions', 'fine'等
            labels: [B] 标签
            weights: 不同尺度的权重

        Returns:
            loss: 标量损失值
        """
        if weights is None:
            weights = {'global': 0.3, 'regions': 0.5, 'fine': 0.2}

        total_loss = 0

        # 1. 全局特征对比
        if 'global' in features_dict:
            global_feat = features_dict['global'].unsqueeze(1)  # [B, 1, D]
            global_loss = self._compute_loss(global_feat, labels)
            total_loss += weights.get('global', 0.3) * global_loss

        # 2. 区域特征对比
        if 'regions' in features_dict and features_dict['regions']:
            region_feats = torch.stack(features_dict['regions'], dim=1)  # [B, num_regions, D]
            region_loss = self._compute_loss(region_feats, labels)
            total_loss += weights.get('regions', 0.5) * region_loss

        # 3. 细粒度特征对比
        if 'fine' in features_dict and features_dict['fine']:
            fine_feats = torch.stack(features_dict['fine'], dim=1)  # [B, num_fine, D]
            fine_loss = self._compute_loss(fine_feats, labels)
            total_loss += weights.get('fine', 0.2) * fine_loss

        return total_loss

    def _compute_loss(self, features, labels):
        """
        计算标准的SupCon损失
        Args:
            features: [B, num_views, D]
            labels: [B]
        """
        num_views = features.shape[1]

        # 展平features
        features = features.reshape(-1, features.shape[-1])  # [B*num_views, D]

        # 计算相似度矩阵
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # 创建mask
        labels = labels.repeat(num_views)
        mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()

        # 去除对角线
        mask.fill_diagonal_(0)

        # 计算损失
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # 只计算正样本对
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        loss = -mean_log_prob_pos.mean()
        return loss


# ============ 使用示例 ============

def integrate_with_vit():
    """
    如何集成到现有的ViT模型中
    修改 modeling_vit_extra_res_pyramid.py
    """

    # 原来的代码:
    # logits = self.classifier(torch.mean(sequence_output, dim=1))

    # 改为:
    # self.ms_classifier = MultiScaleClassificationHead(
    #     hidden_size=config.hidden_size[-1],
    #     num_classes=config.num_labels,
    #     num_regions=4,
    #     fusion_method='attention'
    # )
    # logits = self.ms_classifier(sequence_output)
    pass


def integrate_with_cds():
    """
    如何集成到CDS模块
    修改 cds_modules.py
    """

    # 为每个stage创建多尺度辅助分类器
    # self.ms_auxiliaries = nn.ModuleList([
    #     MultiScaleCDSAuxiliary(
    #         input_dim=config.hidden_size[i],
    #         output_dim=512,
    #         num_regions=4,
    #         stage_idx=i
    #     )
    #     for i in range(len(config.hidden_size))
    # ])

    # 在forward中使用多尺度损失
    # ms_loss = MultiScaleSupConLoss()
    # for stage_idx, stage_features in enumerate(stage_features_list):
    #     ms_features = self.ms_auxiliaries[stage_idx](stage_features)
    #     loss = ms_loss(ms_features, labels)
    pass


if __name__ == "__main__":
    """测试代码"""

    print("Testing Multi-Scale Token Module")
    print("=" * 60)

    # 测试区域提取器
    extractor = RegionTokenExtractor(num_regions=4, use_fine_grain=True)
    sequence = torch.randn(2, 64, 512)  # [B=2, N=64 (8x8), D=512]

    ms_features = extractor(sequence)
    print(f"\n✅ RegionTokenExtractor:")
    print(f"  Global shape: {ms_features['global'].shape}")
    print(f"  Num regions: {len(ms_features['regions'])}")
    print(f"  Region shape: {ms_features['regions'][0].shape}")
    if ms_features['fine']:
        print(f"  Num fine: {len(ms_features['fine'])}")
        print(f"  Fine shape: {ms_features['fine'][0].shape}")

    # 测试分类头
    classifier = MultiScaleClassificationHead(
        hidden_size=512,
        num_classes=10,
        num_regions=4,
        fusion_method='attention'
    )
    logits = classifier(sequence)
    print(f"\n✅ MultiScaleClassificationHead:")
    print(f"  Logits shape: {logits.shape}")

    # 测试CDS辅助分类器
    cds_aux = MultiScaleCDSAuxiliary(
        input_dim=512,
        output_dim=256,
        num_regions=4,
        stage_idx=0
    )
    cds_features = cds_aux(sequence)
    print(f"\n✅ MultiScaleCDSAuxiliary:")
    print(f"  Global normalized: {cds_features['global'].shape}")
    print(f"  Regions count: {len(cds_features['regions'])}")

    # 测试多尺度损失
    ms_loss_fn = MultiScaleSupConLoss()
    labels = torch.tensor([0, 1])
    loss = ms_loss_fn(cds_features, labels)
    print(f"\n✅ MultiScaleSupConLoss:")
    print(f"  Loss value: {loss.item():.4f}")

    print("\n" + "=" * 60)
    print("All tests passed! 🎉")
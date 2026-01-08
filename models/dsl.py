import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class TSSNStream(nn.Module):
    def __init__(self, num_joints=21, in_channels=3, embed_dim=64, disable_layernorm: bool = False):
        super().__init__()
        self.num_joints = num_joints

        norm_shape = nn.LayerNorm(embed_dim * 2) if not disable_layernorm else nn.Identity()
        self.spatial_embed = nn.Sequential(
            nn.Linear(in_channels * num_joints, embed_dim * 2),
            norm_shape,
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.temporal_encoder = nn.Sequential(
            TemporalConv(embed_dim),
            TemporalConv(embed_dim),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        # x shape: (batch, T, V, C) -> V=21 joints
        B, T, V, C = x.shape
        x = x.view(B, T, -1) 
        
        # Spatial Embedding
        x = self.spatial_embed(x) # (B, T, Embed)
        
        # Temporal Modeling (Conv1D requires B, C, T)
        x = x.permute(0, 2, 1) # (B, Embed, T)
        x = self.temporal_encoder(x)
        
        return x # (B, Embed, T)

# 處理位移 
class FTDEStream(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64):
        super().__init__()
        
        self.embed = nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.temporal_encoder = nn.Sequential(
            TemporalConv(embed_dim),
            TemporalConv(embed_dim)
        )
        
        self.energy_gate = nn.Sequential(
            nn.Linear(in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch, T, C) -> only手腕座標
        # 將 (B, T, C) 轉換為 (B, C, T, 1) 以適應 Conv2d
        energy = self.energy_gate(x) # (B, T, 1)
        
        feat = self.embed(x) # (B, T, Embed)
        feat = feat.permute(0, 2, 1) # (B, Embed, T)
        feat = self.temporal_encoder(feat) # (B, Embed, T)
        
        # 強調運動劇烈的幀，抑制靜止幀
        feat = feat * energy.permute(0, 2, 1) 
        
        return feat

# DSLNet 主模型：整合與幾何驅動融合 (Geo-OT)
class DSLNet(nn.Module):
    def __init__(self, num_classes=100, num_joints=21, embed_dim=None, config=None, disable_layernorm: bool = False, **unused_kwargs):
        super().__init__()
        cfg_model = config.get('model', {}) if config else {}

        self.in_channels = 3
        # Webcam / dataset 提供的 traj 為 (x,y,z) 或 (x,y,0) (三維就好)
        self.traj_channels = 3
        self.num_classes = int(num_classes)
        self.num_joints = int(num_joints if num_joints is not None else cfg_model.get('num_joints', 21))
        self.embed_dim = int(embed_dim if embed_dim is not None else cfg_model.get('embed_dim', 64)) 

        self.stream_shape = TSSNStream(
            num_joints=self.num_joints,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            disable_layernorm=disable_layernorm,
        )
        self.stream_traj = FTDEStream(in_channels=self.traj_channels, embed_dim=self.embed_dim)

        # 融合層 (Cross Attention 簡化版 -> Concat + MLP)
        self.fusion_dim = self.embed_dim * 2
        self.attention = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim // 4),
            nn.ReLU(),
            nn.Linear(self.fusion_dim // 4, self.fusion_dim),
            nn.Sigmoid()
        )
        norm_cls = nn.LayerNorm(self.embed_dim) if not disable_layernorm else nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, self.embed_dim),
            norm_cls,
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(self.embed_dim, num_classes)
        )

    

    def forward(self, x_shape, x_traj):
        feat_shape = self.stream_shape(x_shape) # (B, Embed, T)
        feat_traj = self.stream_traj(x_traj)    # (B, Embed, T)
        
        # 時間聚合（把時間序列壓縮為單一向量
        pool_shape = torch.mean(feat_shape, dim=-1) # (B, Embed)
        pool_traj = torch.mean(feat_traj, dim=-1)   # (B, Embed)
        
        # 融合與分類
        combined = torch.cat([pool_shape, pool_traj], dim=1) # (B, Embed*2)
        weights = self.attention(combined) # (B, Embed*2)
        weighted_feat = combined * weights
        out = self.classifier(weighted_feat)
        
        return out, pool_shape, pool_traj
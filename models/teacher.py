import torch
import torch.nn as nn
import torch.nn.functional as F

# 1.處理手部姿態
class TSSNStream(nn.Module):
    def __init__(self, num_joint=21, in_channels=3):
        super(TSSNStream, self).__init__()
        # 論文動態圖卷積(STDGCNN)在這簡化成多層圖卷積塊
        self.stdgcnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.bilstm = nn.LSTM(64 * num_joint, 256, bidirectional=True, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

    def forward(self, x):
        # x shape: (batch, T, V, C) -> V=21 joints
        b, t, v, c = x.shape
        x = x.permute(0, 3, 1, 2) # (B, C, T, V)
        feat = self.stdgcnn(x) 
        feat = feat.permute(0, 2, 3, 1).reshape(b, t, -1)
        lstm_out, _ = self.bilstm(feat)
        f_s, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return f_s

# 2. 處理全局位移 (Facial-centric)
class FTDEStream(nn.Module):
    def __init__(self, in_channels=3):
        super(FTDEStream, self).__init__()
        self.st_conv = nn.Conv1d(in_channels, 128, kernel_size=3, padding=1)
        self.bilstm = nn.LSTM(128, 128, bidirectional=True, batch_first=True)
        
    def forward(self, x):
        # x shape: (batch, T, C) -> only手腕座標
        x = x.transpose(1, 2)
        feat = F.relu(self.st_conv(x)).transpose(1, 2)
        f_t, _ = self.bilstm(feat)
        return f_t

# 3. DSLNet 主模型：整合與幾何驅動融合 (Geo-OT)
class DSLNet(nn.Module):
    def __init__(self, num_classes=100):
        super(DSLNet, self).__init__()
        self.tssn = TSSNStream() 
        self.ftde = FTDEStream() 
        
        self.traj_projection = nn.Linear(256, 512)

        self.geo_projection = nn.Linear(256, 512)
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(512 + 512, 1024),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LSTM):
                for name_p, param in module.named_parameters():
                    if 'weight' in name_p:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name_p:
                        nn.init.constant_(param, 0.0)

    def forward(self, x_shape, x_traj):
        f_s = self.tssn(x_shape) 
        f_t = self.ftde(x_traj)  
        
        f_t_proj = self.traj_projection(f_t)  
        
        f_s_attn, _ = self.cross_attention(f_s, f_t_proj, f_t_proj)
        
        feat_combined = torch.cat([f_s_attn.mean(1), f_t_proj.mean(1)], dim=-1)
        logits = self.classifier(feat_combined)
        
        feat_s_pooled = f_s.mean(1)      
        feat_t_pooled = self.geo_projection(f_t.mean(1))  

        return logits, feat_s_pooled, feat_t_pooled
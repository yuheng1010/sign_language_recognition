import torch.nn as nn
from transformers import VideoMAEForVideoClassification, VideoMAEConfig

class VideoMAE(nn.Module):
    def __init__(self, num_classes=2000):
        super(VideoMAE, self).__init__()
        self.backbone = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Dropout(0.3), # 論文說WLASL100樣本量比較少 so加Dropout助於泛化
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
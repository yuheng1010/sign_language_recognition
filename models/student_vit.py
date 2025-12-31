import torch
import torch.nn as nn
from transformers import MobileViTForImageClassification

class StudentViT(nn.Module):
    """
    學生模型：MobileViT with temporal pooling for video input
    將video輸入 (batch, frames, channels, height, width) 轉換為圖像輸入
    """
    def __init__(self, num_classes=2000, num_frames=16):
        super(StudentViT, self).__init__()
        self.backbone = MobileViTForImageClassification.from_pretrained(
            "apple/mobilevit-small",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        # encoder_layer = nn.TransformerEncoderLayer(d_model=num_classes, nhead=8, batch_first = True)
        # self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        """
        處理video輸入
        Args:
            x: video張量 (batch_size, num_frames, channels, height, width)
        Returns:
            模型輸出
        """
        batch_size, num_frames, c, h, w = x.shape
        x = x.view(batch_size * num_frames, c, h, w)

        frame_outputs = self.backbone(x)
        logits_per_frame = frame_outputs.logits.view(batch_size, num_frames, -1)

        final_logits = logits_per_frame.mean(dim=1)

        class Output:
            def __init__(self, logits):
                self.logits = logits
        return Output(final_logits)
import torch
import torch.nn as nn
import torch.nn.functional as F


class DSLNetWrapper(nn.Module):
    def __init__(self, dslnet_model):
        super(DSLNetWrapper, self).__init__()
        self.model = dslnet_model
        self.requires_skeleton = True
        
    def forward(self, x):
        if isinstance(x, tuple) and len(x) == 2:
            skeleton_shape, skeleton_traj = x
            logits, feat_s, feat_t = self.model(skeleton_shape, skeleton_traj)
            return logits
        else:
            raise TypeError(
                    "DSLNet 需要手部姿態和全局位移輸入 (skeleton_shape, skeleton_traj)"
                    "無法處理video數據 用 WLASLSkeletonDataset代替WLASLDataset"
            )

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha  
        self.ce_loss = nn.CrossEntropyLoss()
        self.kd_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, targets):
        hard_loss = self.ce_loss(student_logits, targets)
        soft_loss = self.kd_loss(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)

        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss

        return total_loss, hard_loss, soft_loss

class DistillationTrainer(nn.Module):
    def __init__(self, teacher_model, student_model, temperature=4.0, alpha=0.5):
        super(DistillationTrainer, self).__init__()
        if hasattr(teacher_model, 'requires_skeleton'):
            print("警告: 教師模型需要手部姿態和全局位移輸入，請確保使用正確的數據集")
            self.requires_skeleton = True
        else:
            self.requires_skeleton = False
            
        self.teacher = teacher_model
        self.student = student_model
        self.kd_loss = KnowledgeDistillationLoss(temperature, alpha)

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x, targets):
        with torch.no_grad():
            teacher_outputs = self.teacher(x)
            
            if isinstance(teacher_outputs, tuple):
                teacher_logits = teacher_outputs[0] # DSLNet (logits, feat_s, feat_t)
            elif hasattr(teacher_outputs, 'logits'):
                teacher_logits = teacher_outputs.logits # VideoMAE
            else:
                teacher_logits = teacher_outputs

        student_outputs = self.student(x)
        
        if hasattr(student_outputs, 'logits'):
            student_logits = student_outputs.logits
        else:
            student_logits = student_outputs

        total_loss, hard_loss, soft_loss = self.kd_loss(
            student_logits, teacher_logits, targets
        )

        return {
            'total_loss': total_loss,
            'hard_loss': hard_loss,
            'soft_loss': soft_loss,
            'student_logits': student_logits,
            'teacher_logits': teacher_logits
        }
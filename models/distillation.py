import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeDistillationLoss(nn.Module):
    """
    知識蒸餾損失函數
    結合硬標籤損失（CrossEntropy）和軟標籤損失（KL Divergence）
    """
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
        self.teacher = teacher_model
        self.student = student_model
        self.kd_loss = KnowledgeDistillationLoss(temperature, alpha)

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x, targets):
        with torch.no_grad():
            teacher_outputs = self.teacher(x)
            teacher_logits = teacher_outputs.logits

        student_outputs = self.student(x)
        student_logits = student_outputs.logits

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
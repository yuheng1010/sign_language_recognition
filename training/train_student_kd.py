import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import yaml
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pro.dataset import WLASLDataset
from models.student_vit import StudentViT
from models.distillation import DistillationTrainer, DSLNetWrapper
from data_pro.sampler import BalancedSampler

try:
    from models.videomae import VideoMAE
    VIDEOMAE_AVAILABLE = True
except ImportError:
    VIDEOMAE_AVAILABLE = False
    print("警告: VideoMAE 模型不可用")
    
try:
    from models.teacher import DSLNet
    DSLNET_AVAILABLE = True
except ImportError:
    DSLNET_AVAILABLE = False
    print("警告: DSLNet 模型不可用")

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_device(device_config):
    if device_config == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_config

def load_teacher_model(model_path, num_classes=2000, model_type='auto'):

    print(f"載入教師模型: {model_path}")

    device_context = torch.device('cpu')
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    if model_type == 'auto':
        if 'dslnet' in model_path.lower() or 'teacher' in model_path.lower():
            model_type = 'dslnet'
        else:
            model_type = 'videomae'

    if model_type == 'dslnet':
        if not DSLNET_AVAILABLE:
            raise ImportError("DSLNet 模型不可用，請確保 models/teacher.py 存在")
        print("使用 DSLNet 教師模型")
        print("警告: DSLNet 需要骨架數據，無法直接用於視頻數據的知識蒸餾")
        print("      如需使用 DSLNet 進行知識蒸餾，請使用 WLASLSkeletonDataset")
        teacher_model = DSLNetWrapper(DSLNet(num_classes=num_classes))
    elif model_type == 'videomae':
        if not VIDEOMAE_AVAILABLE:
            raise ImportError("VideoMAE 模型不可用，請確保 models/videomae.py 存在")
        print("使用 VideoMAE 教師模型")
        teacher_model = VideoMAE(num_classes=num_classes)
    else:
        raise ValueError(f"未知的模型類型: {model_type}")

    if 'model_state_dict' in checkpoint:
        teacher_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        teacher_model.load_state_dict(checkpoint)

    teacher_model = teacher_model.to(device_context)

    for param in teacher_model.parameters():
        param.data = param.data.to(device_context)
        if param.grad is not None:
            param.grad.data = param.grad.data.to(device_context)

    for buffer in teacher_model.buffers():
        buffer.data = buffer.data.to(device_context)

    print("教師模型載入成功!")
    print(f"教師模型類型: {model_type}")
    print(f"教師模型設備: {next(teacher_model.parameters()).device}")
    return teacher_model

def calculate_accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train_student_kd(config_path="configs/student.yaml"):
    print("=" * 60)
    print("開始知識蒸餾訓練 - 學生模型 (MobileViT)")
    print("=" * 60)

    config = load_config(config_path)
    print(f"載入配置: {config_path}")
    device = config.get('device', 'cpu')  
    print(f"使用設備: {device}")

    teacher_model_path = config['model']['teacher_path']
    if not os.path.exists(teacher_model_path):
        print(f"找不到教師模型文件: {teacher_model_path}")
        print("請先運行教師模型訓練")
        return

    print("初始化數據集...")
    train_dataset = WLASLDataset(
        config['data']['json_file'],
        config['data']['root_dir'],
        mode='train',
        num_classes=config['model']['num_classes'],
        num_frames=config['input']['num_frames'],
        img_size=config['input']['img_size']
    )
    val_dataset = WLASLDataset(
        config['data']['json_file'],
        config['data']['root_dir'],
        mode='val',
        num_classes=config['model']['num_classes'],
        num_frames=config['input']['num_frames'],
        img_size=config['input']['img_size']
    )

    train_sampler = BalancedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        num_workers=2, 
        pin_memory=False  # 在cpu訓練不用
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2,  
        pin_memory=False  
    )

    print(f"載入教師模型: {teacher_model_path}")
    teacher_model = load_teacher_model(teacher_model_path, num_classes=config['model']['num_classes'])
    teacher_model = teacher_model.to(device)
    teacher_model.eval()  

    print("初始化學生模型 (MobileViT)...")
    student_model = StudentViT(num_classes=config['model']['num_classes']).to(device)

    print("初始化知識蒸餾訓練器...")
    distillation_trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=config['distillation']['temperature'],
        alpha=config['distillation']['alpha']
    ).to(device)

    optimizer = optim.AdamW(
        student_model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
    )

    best_acc = 0.0
    student_save_path = config['save']['model_path']
    os.makedirs(os.path.dirname(student_save_path), exist_ok=True)

    print(f"開始知識蒸餾訓練 {config['training']['epochs']} 個 epoch...")
    print(f"訓練樣本: {len(train_dataset)} | 驗證樣本: {len(val_dataset)}")
    print(f"目標類別數: {config['model']['num_classes']}")
    print("蒸餾配置: 溫度=4.0, alpha =0.5 (硬標籤:軟標籤 = 1:1)")
    print("-" * 60)

    for epoch in range(config['training']['epochs']):
        distillation_trainer.train()
        running_total_loss = 0.0
        running_hard_loss = 0.0
        running_soft_loss = 0.0

        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        print("-" * 40)

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = distillation_trainer(inputs, labels)

            outputs['total_loss'].backward()
            optimizer.step()

            running_total_loss += outputs['total_loss'].item()
            running_hard_loss += outputs['hard_loss'].item()
            running_soft_loss += outputs['soft_loss'].item()

            if (i + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Batch {i+1}/{len(train_loader)} | "
                      f"Total: {outputs['total_loss'].item():.4f} | "
                      f"Hard: {outputs['hard_loss'].item():.4f} | "
                      f"Soft: {outputs['soft_loss'].item():.4f} | "
                      f"LR: {current_lr:.6f}")

        distillation_trainer.eval()
        val_loss = 0.0
        top1_acc = 0.0
        top5_acc = 0.0
        total_batches = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = distillation_trainer(inputs, labels)

                val_loss += outputs['total_loss'].item()

                acc1, acc5 = calculate_accuracy(outputs['student_logits'], labels, topk=(1, 5))
                top1_acc += acc1.item()
                top5_acc += acc5.item()
                total_batches += 1

        avg_train_total_loss = running_total_loss / len(train_loader)
        avg_train_hard_loss = running_hard_loss / len(train_loader)
        avg_train_soft_loss = running_soft_loss / len(train_loader)
        avg_val_loss = val_loss / total_batches
        avg_top1 = top1_acc / total_batches
        avg_top5 = top5_acc / total_batches

        print(f"訓練完成 | 總損失: {avg_train_total_loss:.4f}")
        print(f"   硬標籤損失: {avg_train_hard_loss:.4f} | 軟標籤損失: {avg_train_soft_loss:.4f}")
        print(f"   驗證損失: {avg_val_loss:.4f}")
        print(f"   Top-1: {avg_top1:.2f}% | Top-5: {avg_top5:.2f}%")

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"學習率已更新為: {current_lr:.6f}")

        if avg_top1 > best_acc:
            best_acc = avg_top1
            torch.save({
                'epoch': epoch + 1,
                'student_model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'config': config
            }, student_save_path)
            print(f"新的最佳學生模型已保存! (Top-1: {best_acc:.2f}%)")
            print(f"保存路徑: {student_save_path}")

    print("\n知識蒸餾訓練完成!")
    print(f"最佳驗證準確度: {best_acc:.2f}%")
    print(f"最終學生模型已保存至: {student_save_path}")

    # 比較教師和學生模型大小
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    compression_ratio = teacher_params / student_params

    print("\n模型壓縮統計:")
    print(f"   教師模型參數: {teacher_params:,}")
    print(f"   學生模型參數: {student_params:,}")
    print(f"   壓縮比例: {compression_ratio:.2f}x")

def main():
    parser = argparse.ArgumentParser(description='知識蒸餾訓練學生模型')
    parser.add_argument('--config', type=str, default='configs/student.yaml',
                       help='配置文件路徑')
    parser.add_argument('--resume', action='store_true',
                       help='從檢查點恢復訓練')

    args = parser.parse_args()

    if args.resume:
        print("從檢查點恢復訓練...")
        # TODO: 實現恢復邏輯

    train_student_kd(args.config)

if __name__ == "__main__":
    main()

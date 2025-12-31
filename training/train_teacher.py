import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import yaml
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pro.dataset import WLASLDataset
from models.videomae import VideoMAE
from data_pro.sampler import BalancedSampler

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

def create_optimizer(model, config):
    optimizer_name = config['optimizer'].get('name', 'adamw').lower()

    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=config['optimizer'].get('betas', [0.9, 0.999]),
            eps=config['optimizer'].get('eps', 1e-8)
        )
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=config['optimizer'].get('betas', [0.9, 0.999]),
            eps=config['optimizer'].get('eps', 1e-8)
        )
    else:
        raise ValueError(f"不支持的優化器: {optimizer_name}")

    return optimizer

def create_scheduler(optimizer, config):
    scheduler_type = config.get('scheduler', {}).get('type', 'cosine').lower()
    warmup_epochs = config['training'].get('warmup_epochs', 0)
    total_epochs = config['training']['epochs']

    if scheduler_type == 'cosine':
        if warmup_epochs > 0:
            # 用帶warmup的cosine annealing
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                else:
                    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                    return 0.5 * (1 + np.cos(np.pi * progress))
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    elif scheduler_type == 'linear':
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_epochs
        )
    else:
        raise ValueError(f"不支持的調度器類型: {scheduler_type}")

    return scheduler

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

def train_teacher(config_path="configs/teacher.yaml"):
    print("=" * 60)
    print("開始訓練教師模型 (VideoMAE)")
    print("=" * 60)

    config = load_config(config_path)
    print(f"載入配置: {config_path}")

    device = get_device(config.get('device', 'auto'))
    config['device'] = device
    print(f"使用設備: {device}")

    print("初始化數據集...")
    train_dataset = WLASLDataset(
        config['data']['json_file'],
        config['data']['root_dir'],
        mode='train',
        num_classes=config['model']['num_classes'],
        num_frames=config['input']['num_frames'],
        img_size=config['input']['img_size'],
        config=config
    )
    val_dataset = WLASLDataset(
        config['data']['json_file'],
        config['data']['root_dir'],
        mode='val',
        num_classes=config['model']['num_classes'],
        num_frames=config['input']['num_frames'],
        img_size=config['input']['img_size'],
        config=config
    )

    train_sampler = BalancedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )

    print("初始化 VideoMAE 教師模型...")
    model = VideoMAE(num_classes=config['model']['num_classes']).to(device)

    # 使用標籤平滑的交叉熵損失
    if config['training'].get('use_label_smoothing', False):
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    best_acc = 0.0
    patience = config.get('validation', {}).get('early_stopping', {}).get('patience', 10)
    patience_counter = 0
    save_dir = os.path.dirname(config['save']['model_path'])
    os.makedirs(save_dir, exist_ok=True)

    print(f"開始訓練 {config['training']['epochs']} 個 epoch...")
    print(f"訓練樣本: {len(train_dataset)} | 驗證樣本: {len(val_dataset)}")
    print(f"目標類別數: {config['model']['num_classes']}")
    print("-" * 60)

    for epoch in range(config['training']['epochs']):
        model.train()
        running_loss = 0.0

        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        print("-" * 40)

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)

            loss.backward()

            if 'gradient_clip' in config['training']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])

            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Batch {i+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | LR: {current_lr:.6f}")

        model.eval()
        val_loss = 0.0
        top1_acc = 0.0
        top5_acc = 0.0
        total_batches = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()

                acc1, acc5 = calculate_accuracy(outputs.logits, labels, topk=(1, 5))
                top1_acc += acc1.item()
                top5_acc += acc5.item()
                total_batches += 1

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / total_batches
        avg_top1 = top1_acc / total_batches
        avg_top5 = top5_acc / total_batches

        print(f"訓練完成 | 平均訓練損失: {avg_train_loss:.4f}")
        print(f"   驗證損失: {avg_val_loss:.4f}")
        print(f"   Top-1: {avg_top1:.2f}% | Top-5: {avg_top5:.2f}%")

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"學習率已更新為: {current_lr:.6f}")

        if avg_top1 > best_acc:
            best_acc = avg_top1
            patience_counter = 0  # 重置 patience 計數器

            teacher_save_path = config['save']['model_path']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'config': config
            }, teacher_save_path)
            print(f"新的最佳模型已保存! (Top-1: {best_acc:.2f}%)")
            print(f"保存路徑: {teacher_save_path}")
        else:
            patience_counter += 1
            print(f"驗證準確度未提升 ({patience_counter}/{patience})")

        # 早停檢查
        if patience_counter >= patience:
            print(f"早停觸發！{patience} 個 epoch 內無改善")
            break

    print("\n教師模型訓練完成!")
    print(f"最佳驗證準確度: {best_acc:.2f}%")
    print(f"最終模型已保存至: {teacher_save_path}")

def main():
    parser = argparse.ArgumentParser(description='訓練教師模型 (VideoMAE)')
    parser.add_argument('--config', type=str, default='configs/teacher.yaml',
                       help='配置文件路徑')
    parser.add_argument('--resume', action='store_true',
                       help='從檢查點恢復訓練')

    args = parser.parse_args()

    if args.resume:
        print("從檢查點恢復訓練...")
        # TODO: 實現恢復邏輯

    train_teacher(args.config)

if __name__ == "__main__":
    main()

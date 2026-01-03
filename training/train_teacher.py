import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
import yaml
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pro.dataset import WLASLSkeletonDataset 
from models.teacher import DSLNet 
from data_pro.sampler import BalancedSampler

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_device(device_config):
    if device_config == "auto":
        if torch.cuda.is_available(): return "cuda"
        if torch.backends.mps.is_available(): return "mps"
        return "cpu"
    return device_config

def calculate_accuracy(output, target, topk=(1, 5, 10)):
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

def geometric_consistency_loss(f_s, f_t):
    return 1 - F.cosine_similarity(f_s, f_t).mean()

def train_teacher(config_path="configs/teacher.yaml", resume=False):
    print("=" * 60)
    print("開始訓練 DSLNet model")
    print("=" * 60)

    config = load_config(config_path)
    device = get_device(config.get('device', 'auto'))
    
    print("初始化骨架數據集...")
    train_dataset = WLASLSkeletonDataset(config['data']['json_file'], mode='train', config=config)

    try:
        val_dataset = WLASLSkeletonDataset(config['data']['json_file'], mode='val', config=config)
        if len(val_dataset.samples) == 0 or val_dataset.use_mock_data:
            print("驗證集沒有真實骨架數據，將使用模擬數據進行驗證")
            print("訓練將使用全部真實骨架數據")
        else:
            print("驗證集使用真實骨架數據")
    except Exception as e:
        print(f"創建驗證集失敗: {e}，將使用模擬數據進行驗證")

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                              sampler=BalancedSampler(train_dataset), num_workers=0)  
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)

    print(f"訓練樣本: {len(train_dataset)} | 驗證樣本: {len(val_dataset)}")

    model = DSLNet(num_classes=config['model']['num_classes']).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    alpha_geo = config['training'].get('alpha_geo', 0.1) # L = L_CE + alpha * L_geo 
    def geometric_consistency_loss(f_s, f_t):
        """
        calculate geometric consistency loss
        f_s: (B, 512) shape feature (projected)
        f_t: (B, 512) trajectory feature (projected)
        """
        return 1 - F.cosine_similarity(f_s, f_t, dim=-1).mean()

    best_acc = 0.0
    start_epoch = 0

    if resume:
        checkpoint_path = config['save']['model_path']
        if os.path.exists(checkpoint_path):
            print(f"載入檢查點: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_acc = checkpoint.get('best_acc', 0.0)
            print(f"模型狀態已恢復 | 從第 {start_epoch} epoch 繼續")
            print(f"當前最佳準確率: {best_acc:.2f}%")
        else:
            print(f"checkpoint不存在: {checkpoint_path}，從頭開始訓練")
    patience = config.get('validation', {}).get('early_stopping', {}).get('patience', 10)
    patience_counter = 0
    save_dir = os.path.dirname(config['save']['model_path'])
    os.makedirs(save_dir, exist_ok=True)

    print(f"開始訓練 {config['training']['epochs']} 個 epoch...")
    print(f"訓練樣本: {len(train_dataset)} | 驗證樣本: {len(val_dataset)}")
    print(f"目標類別數: {config['model']['num_classes']}")
    print("-" * 60)

    for epoch in range(start_epoch, config['training']['epochs']):
        model.train()
        running_loss = 0.0
        
        for i, (skeleton_shape, skeleton_traj, labels) in enumerate(train_loader):
            # DSLNet 需要形態與軌跡兩種輸入
            x_s, x_t, labels = skeleton_shape.to(device), skeleton_traj.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, feat_s, feat_t = model(x_s, x_t) 
            
            loss_ce = ce_criterion(outputs, labels)
            loss_geo = geometric_consistency_loss(feat_s, feat_t)

            total_loss = loss_ce + alpha_geo * loss_geo
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += total_loss.item()

        model.eval()
        top1_acc, top5_acc, top10_acc = 0.0, 0.0, 0.0
        with torch.no_grad():
            for x_s, x_t, labels in val_loader:
                x_s, x_t, labels = x_s.to(device), x_t.to(device), labels.to(device)
                outputs, _, _ = model(x_s, x_t)
                
                acc1, acc5, acc10 = calculate_accuracy(outputs, labels, topk=(1, 5, 10))
                top1_acc += acc1.item()
                top5_acc += acc5.item()
                top10_acc += acc10.item()

        avg_top1 = top1_acc / len(val_loader)
        avg_top5 = top5_acc / len(val_loader)
        avg_top10 = top10_acc / len(val_loader)
        print(f"Epoch {epoch+1} | Loss: {running_loss/len(train_loader):.4f} | Top-1: {avg_top1:.2f}% | Top-5: {avg_top5:.2f}% | Top-10: {avg_top10:.2f}%")

        if avg_top1 > best_acc:
            best_acc = avg_top1
            torch.save({'model_state_dict': model.state_dict(), 'best_acc': best_acc}, config['save']['model_path'])
            print(f"儲存最佳模型: {best_acc:.2f}%")

        scheduler.step()

    print(f"訓練完成！最佳準確度: {best_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='訓練 DSLNet 教師模型')
    parser.add_argument('--config', type=str, default='configs/teacher.yaml',
                       help='配置文件路徑')
    parser.add_argument('--resume', action='store_true',
                       help='從檢查點恢復訓練')
    
    args = parser.parse_args()
    train_teacher(config_path=args.config, resume=args.resume)
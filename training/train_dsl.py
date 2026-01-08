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
import csv
import math
import copy
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from typing import Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pro.dataset import WLASLSkeletonDataset 
from models.dsl import DSLNet 
from data_pro.sampler import BalancedSampler

def export_vocab_csv(dataset: WLASLSkeletonDataset, out_path: str):
    """
    Export training vocabulary (class index -> gloss) to CSV.
    Also includes per-class sample counts from dataset.samples when available.
    """
    if not out_path:
        return

    label_map = getattr(dataset, 'label_map', None) or {}
    idx_to_gloss = {}
    for gloss, idx in label_map.items():
        try:
            idx_to_gloss[int(idx)] = str(gloss)
        except Exception:
            continue

    num_classes = int(getattr(dataset, 'config', {}).get('model', {}).get('num_classes', 0) or 0)
    if num_classes <= 0 and idx_to_gloss:
        num_classes = max(idx_to_gloss.keys()) + 1

    # Count samples per class (best-effort)
    counts = {}
    samples = getattr(dataset, 'samples', None)
    if isinstance(samples, list):
        for s in samples:
            if isinstance(s, dict) and 'label' in s:
                counts[int(s['label'])] = counts.get(int(s['label']), 0) + 1
            elif isinstance(s, (tuple, list)) and len(s) >= 2:
                try:
                    counts[int(s[1])] = counts.get(int(s[1]), 0) + 1
                except Exception:
                    pass

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["class_idx", "gloss", "train_samples"])
        for i in range(num_classes if num_classes > 0 else len(idx_to_gloss)):
            w.writerow([i, idx_to_gloss.get(i, f"class_{i}"), counts.get(i, 0)])
    print(f"✅ 已輸出詞彙 CSV: {out_path}")

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_device(device_config):
    if device_config == "auto":
        if torch.cuda.is_available(): return "cuda"
        if torch.backends.mps.is_available(): return "mps"
        return "cpu"
    return device_config

def mixup_data(x_s: torch.Tensor, x_t: torch.Tensor, y: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply MixUp on both streams with the same lambda."""
    if alpha <= 0:
        return x_s, x_t, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = y.size(0)
    index = torch.randperm(batch_size, device=y.device)
    x_s_mix = lam * x_s + (1 - lam) * x_s[index, :]
    x_t_mix = lam * x_t + (1 - lam) * x_t[index, :]
    y_a, y_b = y, y[index]
    return x_s_mix, x_t_mix, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam: float):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

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

def evaluate_global(model: nn.Module, loader: DataLoader, device: str, topk=(1, 5, 10), max_batches: int = 0):
    """
    Compute global top-k accuracy by counting correct predictions over all samples.
    This is more stable than averaging per-batch percentages when the last batch is small.
    """
    model.eval()
    correct = {k: 0 for k in topk}
    total = 0
    with torch.no_grad():
        for bi, (x_s, x_t, labels) in enumerate(loader):
            if max_batches and bi >= max_batches:
                break
            x_s, x_t, labels = x_s.to(device), x_t.to(device), labels.to(device)
            outputs, _, _ = model(x_s, x_t)
            maxk = max(topk)
            _, pred = outputs.topk(maxk, 1, True, True)  # (B, maxk)
            pred = pred.t()  # (maxk, B)
            total += labels.size(0)
            for k in topk:
                correct[k] += pred[:k].eq(labels.view(1, -1)).any(dim=0).sum().item()
    if total == 0:
        return {k: 0.0 for k in topk}
    return {k: 100.0 * correct[k] / total for k in topk}

def geometric_consistency_loss(f_s, f_t):
    return 1 - F.cosine_similarity(f_s, f_t).mean()

def _build_cosine_with_warmup(optimizer, total_epochs: int, warmup_epochs: int):
    """
    Linearly warm up LR then follow cosine decay. Step once per epoch.
    """
    warmup_epochs = max(0, min(warmup_epochs, total_epochs - 1))
    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def _clone_for_ema(model: nn.Module) -> nn.Module:
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema

@torch.no_grad()
def _update_ema(model: nn.Module, ema_model: nn.Module, decay: float):
    one_minus = 1.0 - decay
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=one_minus)
    for ema_buf, buf in zip(ema_model.buffers(), model.buffers()):
        ema_buf.copy_(buf)

def _update_bn_two_stream(loader: DataLoader, model: nn.Module, device: str):
    """
    BN 重估，支援需要兩路輸入 (x_shape, x_traj) 的 DSLNet。
    參考 torch.optim.swa_utils.update_bn，但手動餵兩個張量。
    """
    if not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        return

    momenta = {}
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.running_mean.zero_()
            module.running_var.fill_(1)
            momenta[module] = module.momentum
            module.momentum = None
            module.num_batches_tracked.zero_()

    n = 0
    with torch.no_grad():
        for x_s, x_t, _ in loader:
            x_s = x_s.to(device)
            x_t = x_t.to(device)
            b = x_s.size(0)
            momentum = b / float(n + b)
            for bn in momenta.keys():
                bn.momentum = momentum
            model(x_s, x_t)
            n += b

    for bn, mom in momenta.items():
        bn.momentum = mom

def train_teacher(config_path="configs/teacher.yaml", resume=False):
    print("=" * 60)
    print("開始訓練 DSLNet model")
    print("=" * 60)

    config = load_config(config_path)
    device = get_device(config.get('device', 'auto'))
    
    print("初始化骨架數據集...")
    train_dataset = WLASLSkeletonDataset(config['data']['json_file'], mode='train', config=config)
    # Export vocabulary CSV (once at startup)
    vocab_csv_path = config.get('save', {}).get('vocab_csv_path', None)
    if not vocab_csv_path:
        # default next to model checkpoint
        vocab_csv_path = os.path.join(os.path.dirname(config['save']['model_path']), "vocab.csv")
    export_vocab_csv(train_dataset, vocab_csv_path)

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
                              sampler=BalancedSampler(
                                  train_dataset,
                                  strategy=config.get('training', {}).get('sampler_strategy', 'even')
                              ) if config.get('training', {}).get('use_balanced_sampler', True) else None,
                              shuffle=not config.get('training', {}).get('use_balanced_sampler', True),
                              num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)

    print(f"訓練樣本: {len(train_dataset)} | 驗證樣本: {len(val_dataset)}")

    model = DSLNet(num_classes=config['model']['num_classes'], config=config).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )
    warmup_epochs = int(config.get('training', {}).get('warmup_epochs', 0))
    scheduler = _build_cosine_with_warmup(
        optimizer,
        total_epochs=config['training']['epochs'],
        warmup_epochs=warmup_epochs
    )
    
    label_smoothing = float(config.get('training', {}).get('label_smoothing', 0.1))
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    alpha_geo = float(config['training'].get('alpha_geo', 0.1)) # L = L_CE + alpha * L_geo
    def geometric_consistency_loss(f_s, f_t):
        """
        calculate geometric consistency loss
        f_s: (B, 512) shape feature (projected)
        f_t: (B, 512) trajectory feature (projected)
        """
        return 1 - F.cosine_similarity(f_s, f_t, dim=-1).mean()

    mixup_alpha = float(config.get('training', {}).get('mixup_alpha', 0.0))
    if mixup_alpha > 0:
        print(f"MixUp 啟用 (alpha={mixup_alpha})")

    swa_cfg = config.get('training', {}).get('swa', {}) or {}
    use_swa = bool(swa_cfg.get('enabled', False))
    swa_start = int(swa_cfg.get('start_epoch', config['training']['epochs'] - 20))
    swa_anneal_epochs = int(swa_cfg.get('anneal_epochs', 10))
    swa_lr = float(swa_cfg.get('lr', config['training']['learning_rate']))
    swa_scheduler = None
    swa_model = None
    swa_started = False
    if use_swa:
        print(f"SWA 啟用 (start={swa_start}, swa_lr={swa_lr}, anneal_epochs={swa_anneal_epochs})")

    use_ema = bool(config.get('training', {}).get('use_ema', False))
    ema_decay = float(config.get('training', {}).get('ema_decay', 0.999))
    ema_model = _clone_for_ema(model) if use_ema else None
    if use_ema:
        print(f"EMA 啟用 (decay={ema_decay})")

    best_acc = 0.0
    start_epoch = 0

    if resume:
        checkpoint_path = config['save']['model_path']
        if os.path.exists(checkpoint_path):
            print(f"載入檢查點: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            if checkpoint.get('optimizer_state_dict'):
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if use_ema and checkpoint.get('ema_state_dict'):
                ema_model.load_state_dict(checkpoint['ema_state_dict'])
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

            optimizer.zero_grad(set_to_none=True)
            if mixup_alpha > 0:
                x_s, x_t, y_a, y_b, lam = mixup_data(x_s, x_t, labels, mixup_alpha)
                outputs, feat_s, feat_t = model(x_s, x_t)
                loss_ce = mixup_criterion(ce_criterion, outputs, y_a, y_b, lam)
            else:
                outputs, feat_s, feat_t = model(x_s, x_t)
                loss_ce = ce_criterion(outputs, labels)

            loss_geo = geometric_consistency_loss(feat_s, feat_t)

            total_loss = loss_ce + alpha_geo * loss_geo
            
            total_loss.backward()
            clip = float(config.get('training', {}).get('gradient_clip', 1.0))
            if clip and clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            if use_ema:
                _update_ema(model, ema_model, ema_decay)
            if use_swa and epoch >= swa_start:
                if not swa_started:
                    swa_model = AveragedModel(model)
                    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr, anneal_epochs=swa_anneal_epochs)
                    swa_started = True
                    print(f"➡️  SWA 已啟動 (epoch {epoch+1})")
                swa_model.update_parameters(model)
                if swa_scheduler:
                    swa_scheduler.step()
            running_loss += total_loss.item()

        # Eval (global accuracy)
        eval_train_batches = int(config.get('training', {}).get('eval_train_batches', 0) or 0)
        eval_model = ema_model if (use_ema and ema_model is not None) else model
        if use_swa and swa_model is not None and swa_started:
            eval_model = swa_model
        tr = evaluate_global(eval_model, train_loader, device, topk=(1, 5, 10), max_batches=eval_train_batches)
        va = evaluate_global(eval_model, val_loader, device, topk=(1, 5, 10))

        print(
            f"Epoch {epoch+1} | Loss: {running_loss/len(train_loader):.4f} | "
            f"Train Top-1: {tr[1]:.2f}% | Val Top-1: {va[1]:.2f}% | "
            f"Val Top-5: {va[5]:.2f}% | Val Top-10: {va[10]:.2f}%"
        )

        avg_top1, avg_top5, avg_top10 = va[1], va[5], va[10]

        if avg_top1 > best_acc:
            best_acc = avg_top1
            torch.save({
                'model_state_dict': (ema_model.state_dict() if use_ema and ema_model is not None else model.state_dict()),
                'ema_state_dict': ema_model.state_dict() if use_ema and ema_model is not None else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'epoch': epoch + 1,
            }, config['save']['model_path'])
            print(f"儲存最佳模型: {best_acc:.2f}% (使用{'EMA' if use_ema else '原始'}權重)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停: {patience} 個 epoch 無提升")
                break

        # Scheduler: cosine until SWA starts; SWA scheduler already stepped per batch
        if not (use_swa and swa_started):
            scheduler.step()

    # Final SWA BN update for best stability
    if use_swa and swa_model is not None and swa_started:
        print("更新 SWA BatchNorm 統計...")
        _update_bn_two_stream(train_loader, swa_model, device=device)

    print(f"訓練完成！最佳準確度: {best_acc:.2f}%")
    print(f"使用以下命令評估測試集:")
    print(f"   python evaluate_test.py --config {args.config}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='訓練 DSLNet 教師模型')
    parser.add_argument('--config', type=str, default='configs/teacher.yaml',
                       help='配置文件路徑')
    parser.add_argument('--resume', action='store_true',
                       help='從檢查點恢復訓練')
    
    args = parser.parse_args()
    train_teacher(config_path=args.config, resume=args.resume)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import yaml
import argparse
from typing import Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pro.dataset import WLASLSkeletonDataset
from models.dsl import DSLNet

from models.prune_dsl import ModelPruner
from training.structured_prune_model import StructuredSpec, structured_prune_dslnet

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

def _infer_dslnet_kwargs_from_state_dict(state_dict) -> dict:
    """Infer embed_dim/num_joints from checkpoint weights (supports packed params)."""
    def _shape_from(k: str):
        v = state_dict.get(k, None)
        if torch.is_tensor(v):
            return tuple(v.shape)
        if isinstance(v, tuple) and len(v) > 0 and torch.is_tensor(v[0]):
            return tuple(v[0].shape)
        return None

    embed_dim = None
    num_joints = None

    s = _shape_from("classifier.0.weight")
    if s and len(s) == 2:
        embed_dim = int(s[0])

    if embed_dim is None:
        s = _shape_from("stream_shape.spatial_embed.4.weight")
        if s and len(s) == 2:
            embed_dim = int(s[0])

    s = _shape_from("stream_shape.spatial_embed.0.weight")
    if s and len(s) == 2:
        num_joints = max(1, int(s[1]) // 3)

    return {
        "embed_dim": int(embed_dim) if embed_dim is not None else None,
        "num_joints": int(num_joints) if num_joints is not None else None,
    }


def load_model(model_path: str, num_classes: int = 100, embed_dim: int = 64, num_joints: int = 21) -> nn.Module:
    print(f"載入DSLNet模型: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint

    inferred = _infer_dslnet_kwargs_from_state_dict(state_dict)

    cfg_embed = embed_dim
    cfg_joints = num_joints
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        cfg_embed = checkpoint['config'].get('model', {}).get('embed_dim', cfg_embed)
        cfg_joints = checkpoint['config'].get('model', {}).get('num_joints', cfg_joints)

    final_embed = inferred["embed_dim"] or cfg_embed
    final_joints = inferred["num_joints"] or cfg_joints

    config = {'model': {'embed_dim': final_embed, 'num_joints': final_joints, 'num_classes': num_classes}}
    model = DSLNet(num_classes=num_classes, num_joints=final_joints, config=config)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    print(f"DSLNet模型載入成功! (embed_dim={model.embed_dim}, num_joints={final_joints})")
    return model

def evaluate_model(model: nn.Module, val_loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    model = model.to(device)  # 確保模型在正確的設備上
    top1_acc = 0.0
    top5_acc = 0.0
    total_batches = 0

    with torch.no_grad():
        for skeleton_shape, skeleton_traj, labels in val_loader:
            labels = labels.to(device)

            # DSLNet 需要兩個輸入
            skeleton_shape, skeleton_traj = skeleton_shape.to(device), skeleton_traj.to(device)
            outputs = model(skeleton_shape, skeleton_traj)

            # DSLNet 返回 (logits, feat_s_pooled, feat_t_pooled)，我們只需要 logits
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            acc1, acc5 = calculate_accuracy(logits, labels, topk=(1, 5))
            top1_acc += acc1.item()
            top5_acc += acc5.item()
            total_batches += 1

    return top1_acc / total_batches, top5_acc / total_batches

def fine_tune_pruned_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                          epochs: int = 10, lr: float = 1e-5, device: str = 'cpu'):
    print(f"開始微調剪枝模型 {epochs} 個 epoch...")

    # Ensure model is on the requested device
    model = model.to(device)
    # 每次微調都重新創建優化器，以避免修剪後的梯度計算圖問題
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (skeleton_shape, skeleton_traj, labels) in enumerate(train_loader):
            labels = labels.to(device)

            # DSLNet 需要兩個輸入
            skeleton_shape, skeleton_traj = skeleton_shape.to(device), skeleton_traj.to(device)
            outputs = model(skeleton_shape, skeleton_traj)

            # DSLNet 返回 (logits, feat_s_pooled, feat_t_pooled)，我們只需要 logits
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            remainder = (i + 1) - ((i + 1) // 20) * 20  # avoid % for TensorRT compat
            if remainder == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {i+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        top1_acc, top5_acc = evaluate_model(model, val_loader, device)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | "
              f"Val Top-1: {top1_acc:.2f}% | Val Top-5: {top5_acc:.2f}%")

        scheduler.step()

        if top1_acc > best_acc:
            best_acc = top1_acc

    print(f"微調完成，最佳準確度: {best_acc:.2f}%")
    return best_acc

def main():
    parser = argparse.ArgumentParser(description='DSLNet 模型剪枝腳本')
    parser.add_argument('--config', type=str, default='configs/prune.yaml',
                       help='配置文件路徑')
    parser.add_argument('--model_path', type=str, default=None,
                       help='輸入模型路徑')
    parser.add_argument('--pruning_ratio', type=float, default=None,
                       help='剪枝比例 (0.0-1.0)')
    parser.add_argument('--pruning_method', type=str, default=None,
                       help='剪枝方法')
    parser.add_argument('--iterative', action='store_true',
                       help='使用迭代剪枝')
    parser.add_argument('--iterations', type=int, default=3,
                       help='迭代次數')
    parser.add_argument('--fine_tune', action='store_true',
                       help='是否進行微調')
    parser.add_argument('--fine_tune_epochs', type=int, default=5,
                       help='微調輪數')
    parser.add_argument('--output_path', type=str, default=None,
                       help='輸出剪枝模型路徑')

    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config.get('device', 'auto'))
    print(f"使用設備: {device}")

    # DSLNet 使用 100 類
    num_classes = config['model'].get('num_classes', 100)

    # 使用命令行參數覆蓋配置文件
    if args.model_path:
        config['model']['input_path'] = args.model_path
    if args.pruning_ratio:
        config['pruning']['target_sparsity'] = args.pruning_ratio
    if args.pruning_method:
        config['pruning']['global']['pruning_method'] = args.pruning_method
    if args.iterative:
        config['pruning']['iterative']['enabled'] = True
        config['pruning']['iterations'] = args.iterations
    if args.fine_tune:
        config['fine_tune']['enabled'] = True
        config['fine_tune']['epochs'] = args.fine_tune_epochs
    if args.output_path:
        config['save']['pruned_model_path'] = args.output_path

    model_path = config['model']['input_path']
    output_path = config['save']['pruned_model_path']

    print("=" * 60)
    print("DSLNet 模型剪枝")
    print("=" * 60)
    print(f"輸入模型: {model_path}")
    print(f"剪枝比例: {config['pruning']['target_sparsity']}")
    print(f"剪枝方法: {config['pruning']['global']['pruning_method']}")
    print(f"迭代剪枝: {config['pruning']['iterative']['enabled']}")
    print(f"微調: {config['fine_tune']['enabled']}")
    print(f"輸出路徑: {output_path}")
    print("-" * 60)

    try:
        model = load_model(model_path, num_classes=num_classes)
        model = model.to(device)  # 將模型移到正確的設備上
    except FileNotFoundError as e:
        print(f"錯誤: {e}")
        print("請先訓練 DSLNet 教師模型: python main.py --stage teacher")
        return

    print("初始化數據集...")
    # 使用骨架數據集
    train_dataset = WLASLSkeletonDataset(
        config['data']['json_file'],
        mode='train',
        config=config
    )
    val_dataset = WLASLSkeletonDataset(
        config['data']['json_file'],
        mode='val',
        config=config
    )

    print(f"訓練集大小: {len(train_dataset)} 樣本")
    print(f"驗證集大小: {len(val_dataset)} 樣本")
    print(f"總類別數: {config['model'].get('num_classes', 100)}")

    from data_pro.sampler import BalancedSampler
    train_sampler = BalancedSampler(train_dataset, strategy='even')
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                            sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                          shuffle=False, num_workers=0)

    print("評估原始模型...")
    original_top1, original_top5 = evaluate_model(model, val_loader, device)
    print(f"   原始準確度: Top-1 {original_top1:.2f}% | Top-5 {original_top5:.2f}%")

    # Pipeline: structured pruning first, then layer-wise unstructured pruning (if enabled)
    pruning_method = 'structured_layer_wise'
    structured_meta = None
    pruner = None

    # 1) structured pruning (always)
    print("使用 structured pruning（先縮小 dense 模型，檔案大小會變小）")
    struct_cfg = config.get('pruning', {}).get('structured', {})
    target_embed_dim = int(struct_cfg.get('embed_dim', 48))
    spec = StructuredSpec(embed_dim=target_embed_dim)
    num_joints = int(config.get('model', {}).get('num_joints', 21))
    model_cpu = model.cpu()
    model, structured_meta = structured_prune_dslnet(model_cpu, spec, num_joints=num_joints)
    model = model.to(device)
    print(f"structured spec: embed_dim {model.embed_dim}")
    print("structured pruning 完成（尚未微調）")
    print("\n最終剪枝統計（structured）:")
    print("   稀疏度: N/A（structured pruning 透過減少 embed_dim 來壓縮，不是把權重變 0）")

    # 2) then layer-wise unstructured pruning (optional)
    if config.get('pruning', {}).get('layer_wise', {}).get('enabled', False):
        print("\n接續進行 layer-wise (unstructured) 剪枝（在 structured 模型上）")
        pruner = ModelPruner(model, device)
        layer_amounts = config['pruning']['layer_wise']['layers']
        print(f"層配置: {layer_amounts}")

        # 調試：檢查哪些層實際存在
        print("檢查模型層結構:")
        available_layers = []
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                available_layers.append(name)
                if name in layer_amounts:
                    print(f"  ✓ {name}: 將剪枝 {layer_amounts[name]:.1%}")
                else:
                    print(f"  - {name}: 不剪枝")
        print(f"總共 {len(available_layers)} 個有權重的層")

        pruner.layer_wise_pruning(
            layer_amounts=layer_amounts,
            method=config['pruning']['global']['pruning_method']
        )

        # 在修剪後清理梯度並進行一次前向傳播來刷新計算圖
        model.zero_grad()
        model.eval()
        with torch.no_grad():
            for skeleton_shape, skeleton_traj, labels in val_loader:
                skeleton_shape, skeleton_traj = skeleton_shape.to(device), skeleton_traj.to(device)
                _ = model(skeleton_shape, skeleton_traj)
                break

        print("\n最終剪枝統計（structured + layer-wise）:")
        print(f"   目標稀疏度: {config['pruning']['target_sparsity']:.1%}")
        print(f"   實際稀疏度: {pruner.get_model_sparsity():.1%}")
    else:
        print("\n跳過 layer-wise (unstructured) 剪枝（pruning.layer_wise.enabled=false）")

    # 最終微調
    if config['fine_tune']['enabled']:
        # Default to config fine_tune.device when present (CPU tends to be more stable on macOS for attention ops)
        fine_tune_device = config.get('fine_tune', {}).get('device', None) or device
        print(f"\n進行最終微調... (device={fine_tune_device})")
        fine_tune_pruned_model(
            model, train_loader, val_loader,
            epochs=config['fine_tune']['epochs'],
            lr=config['fine_tune']['learning_rate'],
            device=fine_tune_device
        )

    print("評估剪枝後模型...")
    final_top1, final_top5 = evaluate_model(model, val_loader, device)
    print(f"   剪枝後準確度: Top-1 {final_top1:.2f}% | Top-5 {final_top5:.2f}%")

    # 對於 unstructured pruning，需要移除 pruning mask
    if pruner is not None:
        print("正在固定剪枝權重 (移除 Mask)...")
        pruner.remove_pruning_masks()
    
    # 保存剪枝模型
    save_dict = {
        'model_state_dict': model.state_dict(),
        'pruning_config': {
            'method': pruning_method,
            'unstructured_method': config['pruning']['global']['pruning_method'],
            'target_sparsity': config['pruning']['target_sparsity'],
            'actual_sparsity': pruner.get_model_sparsity() if pruner is not None else None,
            'num_classes': num_classes,
            'num_joints': config.get('model', {}).get('num_joints', 21),
            'iterative': config['pruning']['iterative']['enabled'],
            'iterations': config['pruning']['iterations'] if config['pruning']['iterative']['enabled'] else 1,
            'fine_tuned': config['fine_tune']['enabled'],
            'fine_tune_epochs': config['fine_tune']['epochs'] if config['fine_tune']['enabled'] else 0,
        },
        'performance': {
            'original_accuracy': {'top1': original_top1, 'top5': original_top5},
            'final_accuracy': {'top1': final_top1, 'top5': final_top5},
            'accuracy_drop': {'top1': original_top1 - final_top1, 'top5': original_top5 - final_top5},
        },
        'pruning_history': pruner.pruning_history if pruner is not None else [],
        'structured_pruning': structured_meta,
    }

    torch.save(save_dict, output_path)
    print(f"剪枝模型已保存至: {output_path}")

    print("\n剪枝總結報告")
    print("=" * 40)
    print(f"   原始準確度: Top-1 {original_top1:.2f}% | Top-5 {original_top5:.2f}%")
    print(f"   剪枝後準確度: Top-1 {final_top1:.2f}% | Top-5 {final_top5:.2f}%")
    print(f"   準確度變化: Top-1 {final_top1 - original_top1:+.2f}% | Top-5 {final_top5 - original_top5:+.2f}%")
    print(f"   剪枝完大小: {output_path} (MB: {os.path.getsize(output_path) / 1024 / 1024:.2f})")
    if pruner is not None:
        print(f"   模型稀疏度: {pruner.get_model_sparsity():.1%}")
    else:
        print("   模型稀疏度: N/A（structured only）")

    print("\nDSLNet 模型剪枝完成!")

if __name__ == "__main__":
    main()

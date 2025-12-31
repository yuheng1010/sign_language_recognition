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

from data_pro.dataset import WLASLDataset
from models.student_vit import StudentViT
from models.prune_student import ModelPruner, load_pruned_model

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

def load_student_model(model_path: str, num_classes: int = 2000) -> nn.Module:
    print(f"載入學生模型: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    model = StudentViT(num_classes=num_classes)

    if 'student_model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['student_model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print("學生模型載入成功!")
    return model

def evaluate_model(model: nn.Module, val_loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    top1_acc = 0.0
    top5_acc = 0.0
    total_batches = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            acc1, acc5 = calculate_accuracy(outputs.logits, labels, topk=(1, 5))
            top1_acc += acc1.item()
            top5_acc += acc5.item()
            total_batches += 1

    return top1_acc / total_batches, top5_acc / total_batches

def fine_tune_pruned_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                          epochs: int = 5, lr: float = 1e-5, device: str = 'cpu'):
    print(f"開始微調剪枝模型 {epochs} 個 epoch...")

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        top1_acc, top5_acc = evaluate_model(model, val_loader, device)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | "
              f"Top-1: {top1_acc:.2f}% | Top-5: {top5_acc:.2f}%")

        scheduler.step()

        if top1_acc > best_acc:
            best_acc = top1_acc

    print(f"微調完成，最佳準確度: {best_acc:.2f}%")
    return model

def main():
    parser = argparse.ArgumentParser(description='學生模型剪枝工具')
    parser.add_argument('--student_model', type=str, default=None,
                       help='學生模型路徑 (默認: best_videomae_wlasl_student_kd_100.pth)')
    parser.add_argument('--pruning_ratio', type=float, default=0.3,
                       help='全域剪枝比例 (0-1)')
    parser.add_argument('--pruning_method', type=str, default='l1_unstructured',
                       choices=['l1_unstructured', 'l2_unstructured', 'random_unstructured'],
                       help='剪枝方法')
    parser.add_argument('--iterative', action='store_true',
                       help='使用迭代剪枝')
    parser.add_argument('--iterations', type=int, default=3,
                       help='迭代剪枝次數')
    parser.add_argument('--fine_tune', action='store_true',
                       help='剪枝後進行微調')
    parser.add_argument('--fine_tune_epochs', type=int, default=5,
                       help='微調輪數')
    parser.add_argument('--output_path', type=str, default=None,
                       help='輸出剪枝模型路徑')

    args = parser.parse_args()

    config = load_config(args.config)
    print(f"載入配置: {args.config}")


    device = get_device(config.get('device', 'auto'))
    config['device'] = device
    print(f"使用設備: {device}")

    if args.student_model is not None:
        config['model']['input_path'] = args.student_model
    if args.pruning_ratio is not None:
        config['pruning']['target_sparsity'] = args.pruning_ratio
    if args.pruning_method is not None:
        config['pruning']['global']['pruning_method'] = args.pruning_method
    if args.iterations is not None:
        config['pruning']['iterations'] = args.iterations
    if args.fine_tune_epochs is not None:
        config['fine_tune']['epochs'] = args.fine_tune_epochs
    if args.output_path is not None:
        config['model']['output_path'] = args.output_path

    student_model_path = config['model']['input_path']
    output_path = config['model']['output_path']

    print("=" * 60)
    print("學生模型剪枝工具")
    print("=" * 60)
    print(f"學生模型: {student_model_path}")
    print(f"剪枝比例: {config['pruning']['target_sparsity']:.1%}")
    print(f"剪枝方法: {config['pruning']['global']['pruning_method']}")
    print(f"迭代剪枝: {args.iterative or config['pruning']['iterative']['enabled']}")
    print(f"輸出路徑: {output_path}")
    print("-" * 60)

    try:
        student_model = load_student_model(student_model_path, num_classes=config['model']['num_classes'])
    except FileNotFoundError as e:
        print(f"錯誤: {e}")
        print("請先運行知識蒸餾訓練: python training/train_student_kd.py")
        return

    val_dataset = WLASLDataset(
        config['data']['json_file'],
        config['data']['root_dir'],
        mode='val',
        num_classes=config['model']['num_classes'],
        num_frames=config.get('input', {}).get('num_frames', 16),
        img_size=config.get('input', {}).get('img_size', 224)
    )
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=2)


    print("評估原始模型性能...")
    original_top1, original_top5 = evaluate_model(student_model, val_loader, device)
    print(f"   原始準確度: Top-1 {original_top1:.2f}% | Top-5 {original_top5:.2f}%")
    original_params = sum(p.numel() for p in student_model.parameters())
    print(f"原始參數量: {original_params:,}")

    pruner = ModelPruner(student_model, device)

    if args.iterative:
        print(f"使用迭代剪枝，目標稀疏度: {args.pruning_ratio:.1%}, 迭代次數: {args.iterations}")
        pruner.iterative_pruning(
            target_sparsity=args.pruning_ratio,
            iterations=args.iterations,
            method=args.pruning_method
        )
    else:
        print(f"使用全域剪枝，剪枝比例: {args.pruning_ratio:.1%}")
        pruner.global_pruning(amount=args.pruning_ratio, method=args.pruning_method)

    pruner.remove_pruning_masks()

    print("評估剪枝後模型性能...")
    pruned_top1, pruned_top5 = evaluate_model(pruner.model, val_loader, device)
    print(f"   剪枝準確度: Top-1 {pruned_top1:.2f}% | Top-5 {pruned_top5:.2f}%")
    final_sparsity = pruner.get_model_sparsity()
    layer_sparsity = pruner.get_layer_sparsity()

    print(f"最終模型稀疏度: {final_sparsity:.1%}")
    print("各層稀疏度:")
    for layer, sparsity in layer_sparsity.items():
        if sparsity > 0.01:  
            print(".1%")
    if args.fine_tune or config['fine_tune']['enabled']:
        fine_tune_epochs = args.fine_tune_epochs or config['fine_tune']['epochs']
        print(f"開始微調剪枝模型 {fine_tune_epochs} 輪...")

        train_dataset = WLASLDataset(
            config['data']['json_file'],
            config['data']['root_dir'],
            mode='train',
            num_classes=config['model']['num_classes'],
            num_frames=config.get('input', {}).get('num_frames', 16),
            img_size=config.get('input', {}).get('img_size', 224)
        )
        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=2)

        pruner.model = fine_tune_pruned_model(
            pruner.model, train_loader, val_loader,
            epochs=fine_tune_epochs, device=device
        )

        final_top1, final_top5 = evaluate_model(pruner.model, val_loader, device)
        print(f"   微調準確度: Top-1 {final_top1:.2f}% | Top-5 {final_top5:.2f}%")
    else:
        final_top1, final_top5 = pruned_top1, pruned_top5

    metadata = {
        'original_accuracy': {'top1': original_top1, 'top5': original_top5},
        'pruned_accuracy': {'top1': pruned_top1, 'top5': pruned_top5},
        'final_accuracy': {'top1': final_top1, 'top5': final_top5},
        'pruning_config': {
            'ratio': config['pruning']['target_sparsity'],
            'method': config['pruning']['global']['pruning_method'],
            'iterative': args.iterative or config['pruning']['iterative']['enabled'],
            'iterations': config['pruning']['iterations'],
            'fine_tune': args.fine_tune or config['fine_tune']['enabled'],
            'fine_tune_epochs': fine_tune_epochs if (args.fine_tune or config['fine_tune']['enabled']) else None,
        },
        'model_stats': {
            'original_params': original_params,
            'final_sparsity': final_sparsity,
            'layer_sparsity': layer_sparsity,
        }
    }

    pruner.save_pruned_model(args.output_path, metadata)

    print("\n剪枝總結報告")
    print("=" * 40)
    print(f"   原始準確度: Top-1 {original_top1:.2f}% | Top-5 {original_top5:.2f}%")
    print(f"   剪枝準確度: Top-1 {pruned_top1:.2f}% | Top-5 {pruned_top5:.2f}%")
    print(f"   最終準確度: Top-1 {final_top1:.2f}% | Top-5 {final_top5:.2f}%")
    print(f"   參數減少: {(final_sparsity * 100):.1f}%")
    print(f"   準確度變化: Top-1 {final_top1 - original_top1:+.2f}%")
    print(f"   模型已保存至: {output_path}")

    print("\n剪枝任務完成!")

if __name__ == "__main__":
    main()

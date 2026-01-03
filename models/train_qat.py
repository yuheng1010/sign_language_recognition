import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.quantization as quant
from torch.quantization import QuantStub, DeQuantStub
import os
import sys
import yaml
import argparse
import copy


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pro.dataset import WLASLDataset
from models.student_vit import StudentViT
from models.prune_student import load_pruned_model

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

class QuantizableStudentViT(nn.Module):
    def __init__(self, num_classes=2000):
        super(QuantizableStudentViT, self).__init__()

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        from transformers import MobileViTForImageClassification
        self.backbone = MobileViTForImageClassification.from_pretrained(
            "apple/mobilevit-small",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, x):
        x = self.quant(x)
        outputs = self.backbone(x)

        if hasattr(outputs, 'logits'):
            outputs.logits = self.dequant(outputs.logits)

        return outputs

    def fuse_model(self):
        print("fusing model layers...")
        for module_name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) and "conv" in module_name:
                bn_name = module_name.replace("conv", "batch_norm")
                relu_name = module_name.replace("conv", "activation")

                try:
                    torch.quantization.fuse_modules(
                        self, [[module_name, bn_name, relu_name]], inplace=True
                    )
                    print(f"融合: {module_name} + {bn_name} + {relu_name}")
                except:
                    pass  

class QATTrainer:
    def __init__(self, model, train_loader, val_loader, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def prepare_qat(self):
        print("準備QAT...")

        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

        self.model.fuse_model()

        torch.quantization.prepare_qat(self.model, inplace=True)

        print("QAT準備完成")

    def train_qat(self, epochs=10, lr=1e-5):
        print(f"開始QAT {epochs} 個 epoch...")

        optimizer = optim.AdamW(
            [p for n, p in self.model.named_parameters() if 'quant' in n or 'weight' in n],
            lr=lr,
            weight_decay=1e-4
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if (i + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Batch {i+1}/{len(self.train_loader)} | "
                          f"Loss: {loss.item():.4f}")

            avg_loss = running_loss / len(self.train_loader)
            top1_acc, top5_acc = self.evaluate()

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | "
                  f"Val Top-1: {top1_acc:.2f}% | Val Top-5: {top5_acc:.2f}%")

            scheduler.step()

            if top1_acc > best_acc:
                best_acc = top1_acc

        print(f"QAT完成，最佳準確度: {best_acc:.2f}%")
        return best_acc

    def evaluate(self):
        self.model.eval()
        top1_acc = 0.0
        top5_acc = 0.0
        total_batches = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                acc1, acc5 = calculate_accuracy(outputs.logits, labels, topk=(1, 5))
                top1_acc += acc1.item()
                top5_acc += acc5.item()
                total_batches += 1

        return top1_acc / total_batches, top5_acc / total_batches

    def convert_to_quantized(self):
        print("轉換為量化模型...")
        self.model.eval()
        quantized_model = torch.quantization.convert(self.model, inplace=False)

        print("量化模型轉換完成")
        return quantized_model

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

def load_model_for_qat(model_path: str, num_classes: int = 2000, model_class=QuantizableStudentViT) -> nn.Module:
    print(f"載入模型用於 QAT: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    model = model_class(num_classes=num_classes)

    if 'student_model_state_dict' in checkpoint:
        state_dict = checkpoint['student_model_state_dict']

        model_state = model.state_dict()
        for name, param in state_dict.items():
            if name in model_state:
                model_state[name].copy_(param)
            elif 'backbone.' + name in model_state:
                model_state['backbone.' + name].copy_(param)

    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print("模型載入成功!")
    return model

def compare_model_sizes(original_model, quantized_model):
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        torch.save(original_model.state_dict(), f.name)
        original_size = os.path.getsize(f.name)

    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        torch.save(quantized_model.state_dict(), f.name)
        quantized_size = os.path.getsize(f.name)

    os.unlink(f.name)

    compression_ratio = original_size / quantized_size

    print("\n模型大小比較:")
    print(f"   原始模型大小: {original_size:,} bytes ({original_size/1024/1024:.2f} MB)")
    print(f"   量化模型大小: {quantized_size:,} bytes ({quantized_size/1024/1024:.2f} MB)")
    print(f"   壓縮比例: {compression_ratio:.2f}x")
    return compression_ratio

def main():
    parser = argparse.ArgumentParser(description='量化感知訓練 (QAT)')
    parser.add_argument('--config', type=str, default='configs/qat.yaml',
                       help='配置文件路徑')
    parser.add_argument('--model_path', type=str, default=None,
                       help='輸入模型路徑 (如果不指定，將從配置中讀取)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='QAT 訓練輪數 (如果不指定，將從配置中讀取)')
    parser.add_argument('--lr', type=float, default=None,
                       help='學習率 (如果不指定，將從配置中讀取)')
    parser.add_argument('--output_path', type=str, default=None,
                       help='輸出量化模型路徑 (如果不指定，將從配置中讀取)')

    args = parser.parse_args()

    config = load_config(args.config)
    print(f"載入配置: {args.config}")

    device = get_device(config.get('device', 'auto'))
    config['device'] = device
    print(f"使用設備: {device}")

    if args.model_path is not None:
        config['model']['input_path'] = args.model_path
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.output_path is not None:
        config['save']['output_path'] = args.output_path

    model_path = config['model']['input_path']
    output_path = config['save']['output_path']

    print("=" * 60)
    print("量化感知訓練 (QAT)")
    print("=" * 60)
    print(f"輸入模型: {model_path}")
    print(f"訓練輪數: {config['training']['epochs']}")
    print(f"學習率: {config['training']['learning_rate']}")
    print(f"輸出路徑: {output_path}")
    print("-" * 60)

    try:
        model = load_model_for_qat(model_path, num_classes=config['model']['num_classes'])
    except FileNotFoundError as e:
        print(f"錯誤: {e}")
        print("請先訓練學生模型: python training/train_student_kd.py")
        return

    print("初始化數據集...")
    train_dataset = WLASLDataset(
        config['data']['json_file'],
        config['data']['root_dir'],
        mode='train',
        num_classes=config['model']['num_classes'],
        num_frames=config.get('input', {}).get('num_frames', 16),
        img_size=config.get('input', {}).get('img_size', 224)
    )
    val_dataset = WLASLDataset(
        config['data']['json_file'],
        config['data']['root_dir'],
        mode='val',
        num_classes=config['model']['num_classes'],
        num_frames=config.get('input', {}).get('num_frames', 16),
        img_size=config.get('input', {}).get('img_size', 224)
    )

    from data_pro.sampler import BalancedSampler
    train_sampler = BalancedSampler(train_dataset, strategy='even')
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                            sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                          shuffle=False, num_workers=2)

    print("評估原始模型...")
    trainer = QATTrainer(model, train_loader, val_loader, device)
    original_top1, original_top5 = trainer.evaluate()
    print(f"   原始準確度: Top-1 {original_top1:.2f}% | Top-5 {original_top5:.2f}%")
    original_model = copy.deepcopy(model)

    trainer.prepare_qat()
    best_acc = trainer.train_qat(epochs=args.epochs, lr=args.lr)
    quantized_model = trainer.convert_to_quantized()

    print("評估量化模型...")
    quant_trainer = QATTrainer(quantized_model, train_loader, val_loader, device)
    final_top1, final_top5 = quant_trainer.evaluate()
    print(f"   量化準確度: Top-1 {final_top1:.2f}% | Top-5 {final_top5:.2f}%")
    compression_ratio = compare_model_sizes(original_model, quantized_model)

    save_dict = {
        'quantized_model_state_dict': quantized_model.state_dict(),
        'qat_config': {
            'epochs': config['training']['epochs'],
            'lr': config['training']['learning_rate'],
            'input_model': model_path,
        },
        'performance': {
            'original_accuracy': {'top1': original_top1, 'top5': original_top5},
            'final_accuracy': {'top1': final_top1, 'top5': final_top5},
            'accuracy_drop': {'top1': original_top1 - final_top1, 'top5': original_top5 - final_top5},
        },
        'compression': {
            'ratio': compression_ratio,
        }
    }

    torch.save(save_dict, output_path)
    print(f"量化模型已保存至: {output_path}")

    print("\nQAT 總結報告")
    print("=" * 40)
    print(f"   原始準確度: Top-1 {original_top1:.2f}% | Top-5 {original_top5:.2f}%")
    print(f"   量化準確度: Top-1 {final_top1:.2f}% | Top-5 {final_top5:.2f}%")
    print(f"   模型壓縮: {compression_ratio:.2f}x")
    print(f"   準確度變化: Top-1 {final_top1 - original_top1:+.2f}% | Top-5 {final_top5 - original_top5:+.2f}%")

    print("\n量化感知訓練完成!")

if __name__ == "__main__":
    main()

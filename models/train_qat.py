import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization as quant
from torch.utils.data import DataLoader
import os
import sys
import yaml
import argparse
import numpy as np
from typing import Dict, Any, Tuple, Optional
import time
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dsl import DSLNet
from models.prune_dsl import load_pruned_model
from data_pro.dataset import WLASLSkeletonDataset
from data_pro.sampler import BalancedSampler


def load_config(config_path: str) -> Dict[str, Any]:
    """載入配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_device(device_config: str) -> str:
    """獲取設備配置"""
    if device_config == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_config


def calculate_accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> list:
    """計算準確度"""
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


def fuse_modules(model: nn.Module) -> nn.Module:
    """
    融合兼容的模塊以提高量化性能
    Conv2d + BatchNorm2d + ReLU -> ConvReLU2d
    Conv2d + BatchNorm2d -> Conv2d
    Linear + ReLU -> LinearReLU
    """
    print("融合模塊以優化量化...")

    fused_any = False

    try:
        # 對於 TSSNStream - 檢查 stdgcnn
        if hasattr(model, 'tssn') and hasattr(model.tssn, 'stdgcnn'):
            stdgcnn = model.tssn.stdgcnn
            modules_list = list(stdgcnn.named_children())  # 只檢查直接子模塊

            print(f"  檢查 TSSNStream.stdgcnn 結構: {len(modules_list)} 個模塊")
            module_types = []
            for i, (name, module) in enumerate(modules_list):
                module_type = type(module).__name__
                module_types.append(module_type)
                print(f"    {i}: {module_type}")

            # 檢查是否有 Conv2d -> BatchNorm2d -> ReLU 模式
            if (len(modules_list) >= 3 and
                isinstance(modules_list[0][1], nn.Conv2d) and
                isinstance(modules_list[1][1], nn.BatchNorm2d) and
                isinstance(modules_list[2][1], nn.ReLU)):
                try:
                    # 嘗試融合 [Conv2d, BatchNorm2d, ReLU]
                    quant.fuse_modules(stdgcnn, [['0', '1', '2']], inplace=True)
                    print("  ✓ 融合 TSSNStream.stdgcnn (Conv2d+BN+ReLU)")
                    fused_any = True
                except Exception as e:
                    print(f"  ⚠ TSSNStream Conv2d+BN+ReLU 融合失敗: {e}")
                    try:
                        quant.fuse_modules(stdgcnn, [['0', '1']], inplace=True)
                        print("  ✓ 融合 TSSNStream.stdgcnn (Conv2d+BN)")
                        fused_any = True
                    except Exception as e2:
                        print(f"  ⚠ TSSNStream Conv2d+BN 融合也失敗: {e2}")
            elif (len(modules_list) >= 2 and
                  isinstance(modules_list[0][1], nn.Conv2d) and
                  isinstance(modules_list[1][1], nn.BatchNorm2d)):
                try:
                    quant.fuse_modules(stdgcnn, [['0', '1']], inplace=True)
                    print("  ✓ 融合 TSSNStream.stdgcnn (Conv2d+BN)")
                    fused_any = True
                except Exception as e:
                    print(f"  ⚠ TSSNStream Conv2d+BN 融合失敗: {e}")
            else:
                print(f"  ℹ TSSNStream.stdgcnn 沒有標準融合模式: {module_types}")

        # 對於 FTDEStream - 檢查是否有卷積相關層
        if hasattr(model, 'ftde'):
            ftde = model.ftde
            modules_list = list(ftde.named_children())  

            print(f"  檢查 FTDEStream 結構: {len(modules_list)} 個模塊")
            module_types = []
            for i, (name, module) in enumerate(modules_list):
                module_type = type(module).__name__
                module_types.append(module_type)
                print(f"    {i}: {module_type}")

            conv_idx = None
            bn_idx = None
            relu_idx = None

            for i, (name, module) in enumerate(modules_list):
                if isinstance(module, nn.Conv2d) and conv_idx is None:
                    conv_idx = i
                elif isinstance(module, nn.BatchNorm2d) and bn_idx is None:
                    bn_idx = i
                elif isinstance(module, nn.ReLU) and relu_idx is None:
                    relu_idx = i

            if (conv_idx is not None and bn_idx is not None and relu_idx is not None and
                conv_idx < bn_idx < relu_idx):
                try:
                    quant.fuse_modules(ftde, [[str(conv_idx), str(bn_idx), str(relu_idx)]], inplace=True)
                    print("  ✓ 融合 FTDEStream (Conv2d+BN+ReLU)")
                    fused_any = True
                except Exception as e:
                    print(f"  ⚠ FTDEStream Conv2d+BN+ReLU 融合失敗: {e}")
                    try:
                        quant.fuse_modules(ftde, [[str(conv_idx), str(bn_idx)]], inplace=True)
                        print("  ✓ 融合 FTDEStream (Conv2d+BN)")
                        fused_any = True
                    except Exception as e2:
                        print(f"  ⚠ FTDEStream Conv2d+BN 融合失敗: {e2}")
            elif (conv_idx is not None and bn_idx is not None and conv_idx < bn_idx):
                try:
                    quant.fuse_modules(ftde, [[str(conv_idx), str(bn_idx)]], inplace=True)
                    print("  ✓ 融合 FTDEStream (Conv2d+BN)")
                    fused_any = True
                except Exception as e:
                    print(f"  ⚠ FTDEStream Conv2d+BN 融合失敗: {e}")
            else:
                print(f"  ℹ FTDEStream 沒有標準融合模式: {module_types}")

        # 對於分類器 - 查找 Linear + ReLU 模式
        if hasattr(model, 'classifier'):
            classifier = model.classifier
            modules_list = list(classifier.named_children())  

            print(f"  檢查 Classifier 結構: {len(modules_list)} 個模塊")
            for i, (name, module) in enumerate(modules_list):
                print(f"    {i}: {type(module).__name__}")

            fused_pairs = []
            for i in range(len(modules_list) - 1):
                current_module = modules_list[i][1]
                next_module = modules_list[i + 1][1]

                if isinstance(current_module, nn.Linear) and isinstance(next_module, nn.ReLU):
                    fused_pairs.append([str(i), str(i + 1)])
                    print(f"    找到 Linear+ReLU 對: {i} -> {i+1}")

            if fused_pairs:
                try:
                    quant.fuse_modules(classifier, fused_pairs, inplace=True)
                    print(f"  ✓ 融合 Classifier {len(fused_pairs)} 組 Linear+ReLU")
                    fused_any = True
                except Exception as e:
                    print(f"  ⚠ Classifier融合失敗: {e}")
                    print(f"    嘗試的融合對: {fused_pairs}")
            else:
                print("  ℹ Classifier 中沒有找到 Linear+ReLU 模式")

        if fused_any:
            print("模塊融合完成")
        else:
            print("沒有找到可融合的模塊模式")

    except Exception as e:
        print(f"模塊融合過程中出現錯誤: {e}")
        print("繼續進行量化而不融合...")

    return model


def prepare_qat_model(model: nn.Module, config: Dict[str, Any]) -> Tuple[nn.Module, bool]:
    """
    準備量化感知訓練模型
    """
    print("準備量化感知訓練模型...")

    requested_backend = config['quantization'].get('backend', 'fbgemm')

    supported_engines = torch.backends.quantized.supported_engines
    print(f"支持的量化後端: {supported_engines}")
    print(f"當前量化引擎: {torch.backends.quantized.engine}")

    if requested_backend in supported_engines:
        backend = requested_backend
    else:
        priority_backends = ['fbgemm', 'qnnpack', 'onednn']
        backend = None
        for b in priority_backends:
            if b in supported_engines:
                backend = b
                break

        if backend is None:
            backend = supported_engines[0] if supported_engines else 'qnnpack'

    try:
        torch.backends.quantized.engine = backend
        print(f"使用量化後端: {backend}")
    except RuntimeError as e:
        print(f"設置量化後端失敗: {e}")
        print("嘗試使用默認後端...")
    quant_mode = config.get('quantization', {}).get('mode', 'auto')
    has_mha = any(isinstance(m, nn.MultiheadAttention) for m in model.modules())
    if quant_mode == 'auto':
        quant_mode = 'dynamic_qat' if has_mha else ('qat' if config['quantization']['qat']['enabled'] else 'ptq')

    if quant_mode in ('dynamic_qat', 'dynamic'):
        print("使用 dynamic QAT（Linear 權重量化，激活保持 FP32）")

        dynamic_qconfig = getattr(quant, 'default_dynamic_qat_qconfig', None)
        if dynamic_qconfig is None:
            dynamic_qconfig = getattr(quant, 'default_dynamic_qconfig', None)
        if dynamic_qconfig is None:
            raise RuntimeError("此 PyTorch 版本缺少 dynamic quantization qconfig（default_dynamic_qat_qconfig/default_dynamic_qconfig）")

        model.qconfig = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.qconfig = dynamic_qconfig
            else:
                module.qconfig = None
        # MultiheadAttention 以及其子模塊全部禁用量化（避免 convert 後破壞注意力內部結構）
        mha_prefixes = [n for n, m in model.named_modules() if isinstance(m, nn.MultiheadAttention)]
        if mha_prefixes:
            for n, m in model.named_modules():
                if any(n == p or n.startswith(p + ".") for p in mha_prefixes):
                    m.qconfig = None
            print(f"已禁用 MultiheadAttention 及其子模塊量化: {mha_prefixes}")

        print("已對 nn.Linear（排除注意力子模塊）套用 dynamic qconfig")

    elif config['quantization']['qat']['enabled']:
        # QAT 配置  
        qconfig = quant.get_default_qat_qconfig(backend)

        if 'observer' in config['quantization']:
            observer_config = config['quantization']['observer']
            if observer_config['type'] == 'moving_average_min_max':
                qconfig = quant.QConfig(
                    activation=quant.MinMaxObserver.with_args(
                        quant_min=observer_config.get('quant_min', -128),
                        quant_max=observer_config.get('quant_max', 127),
                        dtype=torch.qint8
                    ),
                    weight=quant.default_weight_observer
                )

        model.qconfig = qconfig
        print("應用 QAT 量化配置")
    else:
        # PTQ 配置
        model.qconfig = quant.get_default_qconfig(backend)
        print("應用 PTQ 量化配置")


    if config.get('fusion', {}).get('enabled', True) and quant_mode not in ('dynamic_qat', 'dynamic'):
        model.eval()
        model = fuse_modules(model)
    elif config.get('fusion', {}).get('enabled', True):
        print("dynamic 模式：跳過 fusion（避免破壞權重對應）")

    # 確保模型處於訓練模式 (prepare_qat 需要)
    model.train()
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            module.qconfig = None 

    qat_success = False
    try:
        try:
            model = quant.prepare_qat(model, inplace=True)
        except TypeError:
            model = quant.prepare_qat(model)
        print("QAT 模型準備完成")
        qat_success = True
    except Exception as e:
        print(f"QAT 模型準備失敗: {e}")
        print("降級到標準量化準備...")

        try:
            for module in model.modules():
                if hasattr(module, 'qconfig'):
                    delattr(module, 'qconfig')
            model.qconfig = None

            model.qconfig = quant.get_default_qconfig(backend)
            quant.prepare(model, inplace=True)
            print("使用標準量化準備成功")
            qat_success = False  
        except Exception as e2:
            print(f"量化準備完全失敗: {e2}")
            raise e2

    return model, qat_success

    return model


def calibrate_model(model: nn.Module, calibration_loader: DataLoader, device: str, num_samples: int = 1000):
    """
    校準量化模型
    """
    print(f"開始校準，使用 {min(num_samples, len(calibration_loader.dataset))} 個樣本...")

    model.eval()
    model.to(device)

    calibrated_samples = 0
    with torch.no_grad():
        for batch_idx, (skeleton_shape, skeleton_traj, labels) in enumerate(calibration_loader):
            if calibrated_samples >= num_samples:
                break

            skeleton_shape, skeleton_traj = skeleton_shape.to(device), skeleton_traj.to(device)

            _ = model(skeleton_shape, skeleton_traj)

            calibrated_samples += skeleton_shape.size(0)
            remainder = batch_idx - (batch_idx // 10) * 10  
            if remainder == 0:
                print(f"  校準進度: {calibrated_samples}/{num_samples}")

    print("校準完成")
    return model


def convert_to_quantized_model(model: nn.Module, config: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    轉換為量化模型
    """
    print("轉換為量化模型...")
    model.eval()
    quant_mode = None
    if config is not None:
        quant_mode = config.get('quantization', {}).get('mode', 'auto')

    if quant_mode in ('dynamic_qat', 'dynamic'):
        mapping = quant.get_default_dynamic_quant_module_mappings()
        quantized_model = quant.convert(model, mapping=mapping, inplace=False)
    else:
        quantized_model = quant.convert(model, inplace=False)
    print("量化模型轉換完成")
    return quantized_model


def evaluate_quantized_model(model: nn.Module, val_loader: DataLoader, device: str) -> Dict[str, float]:
    """
    評估量化模型性能
    注意：量化模型只能在 CPU 上運行
    """
    print("評估量化模型...")
    model.eval()
    eval_device = 'cpu'
    model.to(eval_device)

    top1_acc = 0.0
    top5_acc = 0.0
    total_samples = 0

    inference_times = []

    with torch.no_grad():
        for skeleton_shape, skeleton_traj, labels in val_loader:
            skeleton_shape = skeleton_shape.to(eval_device)
            skeleton_traj = skeleton_traj.to(eval_device)
            labels = labels.to(eval_device)

            start_time = time.time()
            outputs = model(skeleton_shape, skeleton_traj)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # DSLNet 返回 (logits, feat_s, feat_t)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            acc1, acc5 = calculate_accuracy(logits, labels, topk=(1, 5))
            top1_acc += acc1.item() * skeleton_shape.size(0)
            top5_acc += acc5.item() * skeleton_shape.size(0)
            total_samples += skeleton_shape.size(0)

    avg_top1 = top1_acc / total_samples
    avg_top5 = top5_acc / total_samples
    avg_inference_time = np.mean(inference_times) * 1000  # ms

    memory_size, disk_size = get_quantized_model_size(model)

    results = {
        'top1_accuracy': avg_top1,
        'top5_accuracy': avg_top5,
        'avg_inference_time_ms': avg_inference_time,
        'model_size_mb': memory_size,  # 使用量化後的大小估計
        'disk_size_mb': disk_size
    }
    return results


def get_model_size(model: nn.Module) -> float:
    """獲取模型大小 (MB)（以 state_dict tensor bytes 估算，兼容量化/packed 權重）"""
    total_bytes = 0
    state = model.state_dict()
    for _, v in state.items():
        if torch.is_tensor(v):
            # quantized tensor element_size() is the underlying dtype size (e.g., int8 -> 1)
            total_bytes += v.numel() * v.element_size()
        elif isinstance(v, (list, tuple)):
            for item in v:
                if torch.is_tensor(item):
                    total_bytes += item.numel() * item.element_size()
    return total_bytes / 1024 / 1024


def get_quantized_model_size(model: nn.Module) -> Tuple[float, float]:
    """
    獲取量化模型的實際大小估計
    返回: (內存大小MB, 磁盤大小估計MB)
    """
    memory_size_mb = get_model_size(model)
    disk_size_mb = memory_size_mb
    return memory_size_mb, disk_size_mb


def _count_quantized_tensors_in_state_dict(model: nn.Module) -> Tuple[int, int]:
    """Return (quantized_tensor_count, tensor_count) from state_dict."""
    q_cnt = 0
    t_cnt = 0
    for _, v in model.state_dict().items():
        if torch.is_tensor(v):
            t_cnt += 1
            if getattr(v, "is_quantized", False) or v.dtype in (torch.qint8, torch.quint8):
                q_cnt += 1
    return q_cnt, t_cnt


def save_quantized_model(model: nn.Module, config: Dict[str, Any], metadata: Dict[str, Any]):
    """
    保存量化模型
    """
    output_path = config['save']['output_path']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': config,
        'quantization_config': config['quantization'],
        'metadata': metadata
    }

    torch.save(save_dict, output_path)
    print(f"量化模型已保存到: {output_path}")

    # 保存最小化版本（只含 state_dict）以便比較磁盤大小
    minimal_path = output_path.replace('.pth', '_int8_state_dict.pth')
    try:
        torch.save(model.state_dict(), minimal_path)
        full_mb = os.path.getsize(output_path) / 1024 / 1024
        mini_mb = os.path.getsize(minimal_path) / 1024 / 1024
        print(f"保存最小 state_dict 到: {minimal_path} | 大小 {mini_mb:.2f} MB（完整包 {full_mb:.2f} MB）")
    except Exception as e:
        print(f"⚠ 保存最小 state_dict 失敗: {e}")

    # 保存 TorchScript 版本 (如果啟用)
    if config['save'].get('save_scripted', False):
        script_path = output_path.replace('.pth', '_int8_scripted.pt')
        scripted_model = torch.jit.script(model)
        # Optimize for inference if available (helps on quantized models too)
        try:
            scripted_model = torch.jit.optimize_for_inference(scripted_model)
        except Exception:
            pass
        torch.jit.save(scripted_model, script_path)
        try:
            script_mb = os.path.getsize(script_path) / 1024 / 1024
            print(f"TorchScript (INT8) 已保存到: {script_path} | 大小 {script_mb:.2f} MB")
        except Exception:
            print(f"TorchScript (INT8) 已保存到: {script_path}")


def train_qat_model(config_path: str = "configs/qat.yaml"):
    """
    執行量化感知訓練
    """
    print("=" * 60)
    print("開始量化感知訓練 (QAT)")
    print("=" * 60)

    config = load_config(config_path)

    config['training']['learning_rate'] = float(config['training']['learning_rate'])
    config['training']['weight_decay'] = float(config['training']['weight_decay'])
    config['training']['batch_size'] = int(config['training']['batch_size'])
    config['training']['epochs'] = int(config['training']['epochs'])

    device_config = config.get('device', 'cpu')
    if device_config == 'auto':
        device = 'cpu'  
        print("量化訓練使用CPU設備")
    else:
        device = device_config

    print(f"使用設備: {device}")
    print(f"訓練配置: epochs={config['training']['epochs']}, "
          f"lr={config['training']['learning_rate']}, "
          f"weight_decay={config['training']['weight_decay']}")

    model_path = config['model']['input_path']
    if not os.path.exists(model_path):
        print(f"找不到模型文件: {model_path}")
        print("請先運行模型剪枝階段")
        return False

    print(f"載入剪枝模型: {model_path}")
    model, pruning_metadata = load_pruned_model(DSLNet, model_path, device)

    sparsity = pruning_metadata.get('model_sparsity', 0.0)
    print(f"剪枝模型載入成功，稀疏度: {sparsity:.1%}")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if 'performance' in checkpoint:
        perf = checkpoint['performance']
        orig_acc = perf.get('original_accuracy', {})
        final_acc = perf.get('final_accuracy', {})
        if orig_acc and final_acc:
            print(f"剪枝前準確度: Top-1 {orig_acc.get('top1', 0):.2f}% | Top-5 {orig_acc.get('top5', 0):.2f}%")
            print(f"剪枝後準確度: Top-1 {final_acc.get('top1', 0):.2f}% | Top-5 {final_acc.get('top5', 0):.2f}%")

    print("初始化數據集...")
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

    calibration_dataset = torch.utils.data.Subset(
        train_dataset,
        np.random.choice(len(train_dataset), min(config['data']['calibration_samples'], len(train_dataset)), replace=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=BalancedSampler(train_dataset),
        num_workers=0
    )
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )

    print(f"訓練樣本: {len(train_dataset)} | 驗證樣本: {len(val_dataset)} | 校準樣本: {len(calibration_dataset)}")

    print("\n評估原始模型...")
    original_results = evaluate_quantized_model(model, val_loader, device)

    model, is_qat = prepare_qat_model(model, config)
    if is_qat:
        print("使用 QAT (Quantization Aware Training)")
    else:
        print("使用標準量化訓練")

    training_mode = "QAT" if is_qat else "量化微調"
    print(f"\n開始 {training_mode} 訓練...")

    lr = config['training']['learning_rate']
    if not is_qat:
        lr = lr * 0.1  
        print(f"使用微調學習率: {lr}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config['training']['weight_decay']
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])

    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc = 0.0

    for epoch in range(config['training']['epochs']):
        model.train()
        running_loss = 0.0

        for i, (skeleton_shape, skeleton_traj, labels) in enumerate(train_loader):
            skeleton_shape = skeleton_shape.to(device)
            skeleton_traj = skeleton_traj.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(skeleton_shape, skeleton_traj)

            # DSLNet 返回 (logits, feat_s, feat_t)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            loss = ce_criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            running_loss += loss.item()

            remainder = (i + 1) - ((i + 1) // 10) * 10  
            if remainder == 0:
                print(f"Epoch {epoch+1}/{config['training']['epochs']} | Batch {i+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}")

        model.eval()
        top1_acc = 0.0
        with torch.no_grad():
            for skeleton_shape, skeleton_traj, labels in val_loader:
                skeleton_shape = skeleton_shape.to(device)
                skeleton_traj = skeleton_traj.to(device)
                labels = labels.to(device)

                outputs = model(skeleton_shape, skeleton_traj)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                acc1, = calculate_accuracy(logits, labels, topk=(1,))
                top1_acc += acc1.item()

        avg_top1 = top1_acc / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {running_loss/len(train_loader):.4f} | Val Top-1: {avg_top1:.2f}%")

        if avg_top1 > best_acc:
            best_acc = avg_top1

        scheduler.step()

    print("\n進行最終校準...")
    model = calibrate_model(model, calibration_loader, device, config['data']['calibration_samples'])

    quant_mode = config.get('quantization', {}).get('mode', 'auto')
    has_mha = any(isinstance(m, nn.MultiheadAttention) for m in model.modules())
    if quant_mode == 'auto':
        quant_mode = 'dynamic' if has_mha else 'qat'

    if quant_mode in ('dynamic', 'dynamic_qat'):
        print("使用 dynamic INT8 導出（quantize_dynamic）...")

        embed_dim = model.embed_dim if hasattr(model, 'embed_dim') else 64
        num_classes = model.num_classes if hasattr(model, 'num_classes') else 100
        
        export_config = {'model': {'embed_dim': embed_dim}}
        export_model = DSLNet(num_classes=num_classes, config=export_config)

        src_state = model.state_dict()
        cleaned_state = {
            k: v for k, v in src_state.items()
            if ('weight_fake_quant' not in k and 'activation_post_process' not in k)
        }
        export_model.load_state_dict(cleaned_state, strict=False)
        print(f"導出模型 embed_dim={embed_dim}, num_classes={num_classes}")
        export_model.eval()

        supported_engines = torch.backends.quantized.supported_engines
        requested_backend = config.get('quantization', {}).get('backend', 'qnnpack')
        engine = requested_backend if requested_backend in supported_engines else None
        if engine is None:
            for cand in ['fbgemm', 'qnnpack', 'onednn']:
                if cand in supported_engines:
                    engine = cand
                    break
        if engine is None:
            engine = supported_engines[0] if supported_engines else 'qnnpack'
        torch.backends.quantized.engine = engine
        print(f"dynamic 量化引擎: {torch.backends.quantized.engine}")

        dyn_qconfig = getattr(quant, 'default_dynamic_qconfig', None)
        if dyn_qconfig is None:
            raise RuntimeError("此 PyTorch 版本缺少 default_dynamic_qconfig，無法進行 dynamic quantization")

        mha_prefixes = [n for n, m in export_model.named_modules() if isinstance(m, nn.MultiheadAttention)]
        qconfig_spec = {}
        for n, m in export_model.named_modules():
            if isinstance(m, nn.Linear) and not any(n == p or n.startswith(p + ".") for p in mha_prefixes):
                qconfig_spec[n] = dyn_qconfig

        quantized_model = quant.quantize_dynamic(
            export_model,
            qconfig_spec=qconfig_spec,
            dtype=torch.qint8,
            inplace=False
        )
    else:
        quantized_model = convert_to_quantized_model(model, config)

    print("\n評估量化模型...")
    quantized_results = evaluate_quantized_model(quantized_model, val_loader, device)
    print(quantized_results)
    print("檢查量化模型狀態...")
    quantized_modules = []
    qat_modules = []
    prepared_modules = []

    for name, module in quantized_model.named_modules():
        module_type = str(type(module))
        if ('quantized' in module_type.lower() or
            'QuantizedConv' in module_type or
            'QuantizedLinear' in module_type):
            quantized_modules.append(name)

        if hasattr(module, 'weight_fake_quant') or hasattr(module, 'activation_post_process'):
            qat_modules.append(name)

        if hasattr(module, 'qconfig') and module.qconfig is not None:
            prepared_modules.append(name)

    total_modules = len(list(quantized_model.named_modules()))
    print(f"總模塊數: {total_modules}")

    q_tensor_cnt, tensor_cnt = _count_quantized_tensors_in_state_dict(quantized_model)
    print(f"state_dict tensors: {tensor_cnt} | 量化 tensors: {q_tensor_cnt}")
    dtype_counter = Counter()
    for _, v in quantized_model.state_dict().items():
        if torch.is_tensor(v):
            dtype_counter[str(v.dtype)] += 1
    if dtype_counter:
        most_common = ", ".join([f"{k}={v}" for k, v in dtype_counter.most_common(5)])
        print(f"state_dict dtype 分佈(Top5): {most_common}")

    if quantized_modules:
        print(f"✓ 發現 {len(quantized_modules)} 個靜態量化模塊: {quantized_modules[:3]}...")
    else:
        print("ℹ 未發現靜態量化模塊（QAT後轉換為量化模塊）")

    if qat_modules:
        print(f"✓ 發現 {len(qat_modules)} 個QAT模塊（量化感知訓練模塊）")

    if prepared_modules:
        print(f"✓ 發現 {len(prepared_modules)} 個量化準備模塊（動態量化）")

    if quantized_modules or q_tensor_cnt > 0:
        print("模型已完成量化（發現 quantized 模塊或 INT8 tensors）")
    elif qat_modules:
        print("模型使用QAT（量化感知訓練，尚未轉換為量化模塊）")
    elif prepared_modules:
        print("模型已準備量化（尚未 convert）")
    else:
        print("模型未檢測到量化配置（convert 可能未生效）")

    if not quantized_modules and not qat_modules and prepared_modules:
        print("ℹ 使用動態量化：權重在推理時動態轉換，不改變模塊類型")
    elif qat_modules and not quantized_modules:
        print("ℹ QAT模式：訓練時模擬量化，推理時需要轉換為量化模型")

    metadata = {
        'original_model_results': original_results,
        'quantized_model_results': quantized_results,
        'qat_training_config': config['training'],
        'pruning_metadata': pruning_metadata,
        'quantization_backend': config['quantization']['backend'],
        'quantization_mode': 'qat' if is_qat else 'ptq_finetune',
        'best_accuracy': best_acc,
        'compression_ratio': original_results['model_size_mb'] / quantized_results['model_size_mb']
    }

    save_quantized_model(quantized_model, config, metadata)

    quant_mode = "QAT" if is_qat else "量化微調"
    print("\n" + "="*60)
    print(f"{quant_mode} 訓練總結")
    print("="*60)
    print(f"原始準確度: Top-1 {original_results['top1_accuracy']:.2f}% | Top-5 {original_results['top5_accuracy']:.2f}% | 大小 {original_results['model_size_mb']:.2f} MB")
    print(f"量化準確度: Top-1 {quantized_results['top1_accuracy']:.2f}% | Top-5 {quantized_results['top5_accuracy']:.2f}% | 大小 {quantized_results['model_size_mb']:.2f} MB (估計)")
    print(f"準確度變化: Top-1 {quantized_results['top1_accuracy'] - original_results['top1_accuracy']:+.2f}% | Top-5 {quantized_results['top5_accuracy'] - original_results['top5_accuracy']:+.2f}%")

    compression_ratio = original_results['model_size_mb'] / quantized_results['model_size_mb']
    if compression_ratio > 1.1:
        print(f"壓縮比例: {compression_ratio:.1f}x 更小")
    else:
        print("注意: PyTorch量化主要節省的是推理內存和延遲，而非參數存儲大小")
        print("      實際壓縮效果在部署到邊緣設備時更明顯")

    print("="*60)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DSLNet 量化感知訓練')
    parser.add_argument('--config', type=str, default='configs/qat.yaml',
                       help='配置文件路徑')
    parser.add_argument('--model_path', type=str,
                       help='輸入模型路徑')
    parser.add_argument('--epochs', type=int,
                       help='訓練輪數')
    parser.add_argument('--lr', type=float,
                       help='學習率')

    args = parser.parse_args()

    if args.model_path or args.epochs or args.lr:
        config = load_config(args.config)
        if args.model_path:
            config['model']['input_path'] = args.model_path
        if args.epochs:
            config['training']['epochs'] = args.epochs
        if args.lr:
            config['training']['learning_rate'] = args.lr

        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_config = f.name

        success = train_qat_model(temp_config)
        os.unlink(temp_config)
    else:
        success = train_qat_model(args.config)

    if success:
        print("量化感知訓練完成!")
    else:
        print("量化感知訓練失敗!")
        sys.exit(1)

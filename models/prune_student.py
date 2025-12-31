import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import copy

class ModelPruner:
    """
    學生模型剪枝
    支持多種剪枝策略和粒度
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.original_state = copy.deepcopy(model.state_dict())
        self.pruning_history = []

    def get_module_weights(self, module_name: str) -> torch.Tensor:
        module = dict(self.model.named_modules())[module_name]
        return module.weight.data

    def calculate_sparsity(self, tensor: torch.Tensor) -> float:
        return (tensor == 0).float().mean().item()

    def global_pruning(self, amount: float = 0.2, method: str = 'l1_unstructured'):
        """
        全域剪枝：對整個模型進行統一剪枝
        Args:
            amount: 剪枝比例 (0-1)
            method: 剪枝方法 ('l1_unstructured', 'l2_unstructured', 'random_unstructured')
        """
        print(f"開始全域剪枝: {method}, 剪枝比例: {amount:.1%}")

        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and module.weight.requires_grad:
                parameters_to_prune.append((module, 'weight'))

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=getattr(prune, method.split('_')[0].upper() + 'Unstructured'),
            amount=amount
        )

        self.pruning_history.append({
            'type': 'global',
            'method': method,
            'amount': amount,
            'sparsity': self.get_model_sparsity()
        })

        print(f"全域剪枝完成，全局稀疏度: {self.get_model_sparsity():.1%}")

    def layer_wise_pruning(self, layer_amounts: Dict[str, float], method: str = 'l1_unstructured'):
        """
        層級剪枝：對指定層進行不同比例的剪枝
        Args:
            layer_amounts: {layer_name: pruning_amount} 字典
            method: 剪枝方法
        """
        print(f"開始層級剪枝: {method}")

        for layer_name, amount in layer_amounts.items():
            if layer_name in dict(self.model.named_modules()):
                module = dict(self.model.named_modules())[layer_name]
                prune.l1_unstructured(module, name='weight', amount=amount)
                print(f"  {layer_name}: 剪枝比例 {amount:.1%}")

        self.pruning_history.append({
            'type': 'layer_wise',
            'method': method,
            'layer_amounts': layer_amounts,
            'sparsity': self.get_model_sparsity()
        })

    def iterative_pruning(self, target_sparsity: float, iterations: int = 3,
                         method: str = 'l1_unstructured', fine_tune_epochs: int = 1):
        """
        迭代剪枝：逐步增加剪枝比例並微調
        Args:
            target_sparsity: 目標稀疏度
            iterations: 迭代次數
            method: 剪枝方法
            fine_tune_epochs: 每次迭代的微調輪數
        """
        print(f"開始迭代剪枝，目標稀疏度: {target_sparsity:.1%}, 迭代次數: {iterations}")

        current_sparsity = 0.0
        for i in range(iterations):
            remaining_sparsity = target_sparsity - current_sparsity
            iteration_amount = remaining_sparsity / (iterations - i)

            print(f"迭代 {i+1}/{iterations}, 當前稀疏度: {current_sparsity:.1%}, "
                  f"本次剪枝比例: {iteration_amount:.1%}")

            self.global_pruning(amount=iteration_amount, method=method)
            current_sparsity = self.get_model_sparsity()

            if fine_tune_epochs > 0:
                print(f"進行 {fine_tune_epochs} 輪微調...")
                # TODO: 實現微調邏輯

        print(f"迭代剪枝完成，最終稀疏度: {current_sparsity:.1%}")

    def get_model_sparsity(self) -> float:
        total_params = 0
        total_zeros = 0

        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                weight = module.weight.data
                total_params += weight.numel()
                total_zeros += (weight == 0).sum().item()

        return total_zeros / total_params if total_params > 0 else 0.0

    def get_layer_sparsity(self) -> Dict[str, float]:
        layer_sparsity = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                sparsity = self.calculate_sparsity(module.weight.data)
                layer_sparsity[name] = sparsity
        return layer_sparsity

    def remove_pruning_masks(self):
        print("移除剪枝掩碼...")
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
        print("剪枝掩碼已移除")

    def restore_original_model(self):
        print("恢復原始模型...")
        self.model.load_state_dict(self.original_state)
        self.pruning_history = []
        print("原始模型已恢復")

    def save_pruned_model(self, path: str, metadata: Optional[Dict] = None):
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'pruning_history': self.pruning_history,
            'model_sparsity': self.get_model_sparsity(),
            'layer_sparsity': self.get_layer_sparsity(),
        }

        if metadata:
            save_dict.update(metadata)

        torch.save(save_dict, path)
        print(f"剪枝模型已保存到: {path}")
        print(f"模型稀疏度: {self.get_model_sparsity():.1%}")

def load_pruned_model(model_class, path: str, device: str = 'cpu') -> Tuple[nn.Module, Dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model = model_class()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    metadata = {
        'pruning_history': checkpoint.get('pruning_history', []),
        'model_sparsity': checkpoint.get('model_sparsity', 0.0),
        'layer_sparsity': checkpoint.get('layer_sparsity', {}),
    }

    print(f"載入剪枝模型: {path}")
    print(f"模型稀疏度: {metadata['model_sparsity']:.1%}")

    return model, metadata

def apply_magnitude_pruning(model: nn.Module, pruning_ratio: float = 0.2):
    """
    移除最小的權重
    """
    print(f"應用幅度剪枝，剪枝比例: {pruning_ratio:.1%}")

    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight.requires_grad:
            weight_abs = torch.abs(module.weight.data)

            threshold = torch.quantile(weight_abs, pruning_ratio)

            mask = weight_abs > threshold

            module.weight.data *= mask.float()

    sparsity = sum((param == 0).float().mean().item() for param in model.parameters()
                   if param.requires_grad) / sum(1 for param in model.parameters() if param.requires_grad)

    print(f"幅度剪枝完成，稀疏度: {sparsity:.1%}")
    return model

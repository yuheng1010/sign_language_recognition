import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import copy

class ModelPruner:
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
        print(f"開始全域pruning: {method}, 剪枝比例: {amount:.1%}")

        parameters_to_prune = []
        named = dict(self.model.named_modules())
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and getattr(module.weight, "requires_grad", False):
                # nn.MultiheadAttention uses out_proj.weight directly inside functional forward;
                # applying torch.nn.utils.prune to out_proj replaces `weight` with a Tensor and can break.
                if isinstance(module, nn.Linear) and name.endswith('.out_proj'):
                    parent_name = name.rsplit('.', 1)[0]
                    parent = named.get(parent_name, None)
                    if isinstance(parent, nn.MultiheadAttention):
                        continue
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

    def layer_wise_pruning(self, layer_amounts: Dict[str, float], method: str = 'layer_wise'):
        print(f"開始layer剪枝: {method}")

        for layer_name, amount in layer_amounts.items():
            if layer_name in dict(self.model.named_modules()):
                module = dict(self.model.named_modules())[layer_name]
                if hasattr(module, 'weight') and module.weight is not None:
                    # Special-case MultiheadAttention.out_proj: do in-place magnitude pruning
                    # to keep `out_proj.weight` as a Parameter (no pruning hooks).
                    if isinstance(module, nn.Linear) and layer_name.endswith('.out_proj'):
                        parent_name = layer_name.rsplit('.', 1)[0]
                        parent = dict(self.model.named_modules()).get(parent_name, None)
                        if isinstance(parent, nn.MultiheadAttention):
                            w = module.weight.data
                            w_abs = torch.abs(w)
                            threshold = torch.quantile(w_abs, float(amount))
                            mask = w_abs > threshold
                            module.weight.data *= mask.float()
                            print(f"  {layer_name}:剪枝比例 {amount:.1%}")
                            continue

                    prune.l1_unstructured(module, name='weight', amount=amount)
                    print(f"  {layer_name}: 剪枝比例 {amount:.1%}")
                else:
                    print(f"  {layer_name}: 跳過 (沒有權重)")
            else:
                print(f"  {layer_name}: 層不存在")

        self.pruning_history.append({
            'type': 'layer_wise',
            'method': method,
            'layer_amounts': layer_amounts,
            'sparsity': self.get_model_sparsity()
        })

    def iterative_pruning(self, target_sparsity: float, iterations: int = 3,
                         method: str = 'l1_unstructured', fine_tune_epochs: int = 1):
        print(f"開始迭代剪枝，目標稀疏度: {target_sparsity:.1%}, 迭代次數: {iterations}")

        current_sparsity = 0.0
        for i in range(iterations):
            remaining_sparsity = target_sparsity - current_sparsity
            iteration_amount = remaining_sparsity / (iterations - i)

            print(f"迭代 {i+1}/{iterations}, 當前稀疏度: {current_sparsity:.1%}, "
                  f"剪枝比例: {iteration_amount:.1%}")

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
        print("removing pruning masks...")
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
        print("pruning masks removed")

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
        print(f"model sparsity: {self.get_model_sparsity():.1%}")

def load_pruned_model(model_class, path: str, device: str = 'cpu') -> Tuple[nn.Module, Dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    embed_dim = 64  
    num_classes = 100  
    
    if isinstance(checkpoint, dict):
        if 'config' in checkpoint:
            embed_dim = checkpoint['config'].get('model', {}).get('embed_dim', embed_dim)
        elif 'structured_pruning' in checkpoint:
            sp = checkpoint.get('structured_pruning', {}) or {}
            if 'new_embed_dim' in sp:
                embed_dim = int(sp['new_embed_dim'])
            elif 'structured_spec' in sp:
                spec = sp.get('structured_spec', {})
                if 'embed_dim' in spec:
                    embed_dim = int(spec['embed_dim'])
        
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if 'stream_shape.spatial_embed.4.bias' in state_dict:
            embed_dim = state_dict['stream_shape.spatial_embed.4.bias'].shape[0]
        if 'classifier.4.weight' in state_dict:
            num_classes = state_dict['classifier.4.weight'].shape[0]
    
    print(f"推斷模型參數: embed_dim={embed_dim}, num_classes={num_classes}")
    
    config = {'model': {'embed_dim': embed_dim}}
    try:
        model = model_class(num_classes=num_classes, config=config)
    except TypeError:
        # Fallback for older model_class that doesn't accept config
        model = model_class(num_classes=num_classes)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)

    metadata = {
        'pruning_history': checkpoint.get('pruning_history', []),
        'model_sparsity': checkpoint.get('model_sparsity', 0.0),
        'layer_sparsity': checkpoint.get('layer_sparsity', {}),
        'structured_pruning': checkpoint.get('structured_pruning', None),
    }

    print(f"loading pruned model: {path}")
    print(f"model sparsity: {metadata['model_sparsity']:.1%}")

    return model, metadata

def apply_magnitude_pruning(model: nn.Module, pruning_ratio: float = 0.2):
    print(f"applying magnitude pruning, pruning ratio: {pruning_ratio:.1%}")

    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight.requires_grad:
            weight_abs = torch.abs(module.weight.data)

            threshold = torch.quantile(weight_abs, pruning_ratio)

            mask = weight_abs > threshold

            module.weight.data *= mask.float()

    sparsity = sum((param == 0).float().mean().item() for param in model.parameters()
                   if param.requires_grad) / sum(1 for param in model.parameters() if param.requires_grad)

    print(f"magnitude pruning completed, sparsity: {sparsity:.1%}")
    return model

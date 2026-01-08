"""
Structured pruning for the new DSLNet architecture.
Supports pruning embed_dim across the model.
"""
import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dsl import DSLNet, TSSNStream, FTDEStream, TemporalConv


@dataclass
class StructuredSpec:
    """Specification for structured pruning - target dimensions."""
    embed_dim: int  # Target embedding dimension (affects both streams)


def _topk_idx(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Return indices of top-k scores (descending), stable sorted."""
    k = int(k)
    if k <= 0:
        raise ValueError("k must be > 0")
    if k >= scores.numel():
        return torch.arange(scores.numel(), device=scores.device)
    _, idx = torch.topk(scores, k=k, largest=True, sorted=True)
    return idx.sort().values


def _l1_neurons(w: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Compute L1 importance score for neurons along specified dimension."""
    dims = list(range(w.ndim))
    dims.remove(dim)
    return w.abs().sum(dim=dims)


def structured_prune_dslnet(
    model: DSLNet,
    spec: StructuredSpec,
    num_joints: int = 21,
) -> Tuple[DSLNet, Dict]:
    """
    Build a smaller DSLNet by reducing embed_dim.
    
    This creates a new model with smaller dimensions and copies the most
    important weights from the original model.
    """
    model.eval().cpu()
    
    old_embed_dim = model.embed_dim
    new_embed_dim = spec.embed_dim
    
    if new_embed_dim >= old_embed_dim:
        print(f"Warning: new_embed_dim ({new_embed_dim}) >= old_embed_dim ({old_embed_dim}), no pruning needed")
        return model, {"structured_spec": spec.__dict__}
    
    # Create config for new model
    new_config = {
        'model': {
            'embed_dim': new_embed_dim,
            'num_joints': num_joints,
        }
    }
    
    # Build new smaller model
    new_model = DSLNet(
        num_classes=model.num_classes,
        config=new_config
    ).eval().cpu()
    
    # --- Compute importance scores and select top-k indices ---
    
    # For stream_shape.spatial_embed (Linear layers)
    # spatial_embed[0]: Linear(in_channels * num_joints, embed_dim * 2)
    # spatial_embed[4]: Linear(embed_dim * 2, embed_dim)
    shape_embed_0: nn.Linear = model.stream_shape.spatial_embed[0]
    shape_embed_4: nn.Linear = model.stream_shape.spatial_embed[4]
    
    # Scores for intermediate hidden layer (embed_dim * 2)
    hidden_scores = _l1_neurons(shape_embed_0.weight.data, dim=0)  # (embed_dim*2,)
    keep_hidden = _topk_idx(hidden_scores, new_embed_dim * 2)
    
    # Scores for output embed_dim
    embed_scores = _l1_neurons(shape_embed_4.weight.data, dim=0)  # (embed_dim,)
    keep_embed = _topk_idx(embed_scores, new_embed_dim)
    
    # --- Copy weights ---
    
    # stream_shape.spatial_embed[0]: all inputs, pruned outputs
    new_model.stream_shape.spatial_embed[0].weight.data.copy_(
        shape_embed_0.weight.data[keep_hidden]
    )
    new_model.stream_shape.spatial_embed[0].bias.data.copy_(
        shape_embed_0.bias.data[keep_hidden]
    )
    
    # stream_shape.spatial_embed[1]: LayerNorm with pruned features
    old_ln1 = model.stream_shape.spatial_embed[1]
    new_ln1 = new_model.stream_shape.spatial_embed[1]
    new_ln1.weight.data.copy_(old_ln1.weight.data[keep_hidden])
    new_ln1.bias.data.copy_(old_ln1.bias.data[keep_hidden])
    
    # stream_shape.spatial_embed[4]: pruned inputs, pruned outputs
    new_model.stream_shape.spatial_embed[4].weight.data.copy_(
        shape_embed_4.weight.data[keep_embed][:, keep_hidden]
    )
    new_model.stream_shape.spatial_embed[4].bias.data.copy_(
        shape_embed_4.bias.data[keep_embed]
    )
    
    # stream_shape.temporal_encoder: Conv1d layers with pruned channels
    for i in [0, 1]:
        old_conv = model.stream_shape.temporal_encoder[i].conv
        new_conv = new_model.stream_shape.temporal_encoder[i].conv
        old_bn = model.stream_shape.temporal_encoder[i].bn
        new_bn = new_model.stream_shape.temporal_encoder[i].bn
        
        # Conv1d: (out_channels, in_channels, kernel_size)
        new_conv.weight.data.copy_(old_conv.weight.data[keep_embed][:, keep_embed])
        if old_conv.bias is not None:
            new_conv.bias.data.copy_(old_conv.bias.data[keep_embed])
        
        # BatchNorm1d
        new_bn.weight.data.copy_(old_bn.weight.data[keep_embed])
        new_bn.bias.data.copy_(old_bn.bias.data[keep_embed])
        new_bn.running_mean.data.copy_(old_bn.running_mean.data[keep_embed])
        new_bn.running_var.data.copy_(old_bn.running_var.data[keep_embed])
    
    # --- stream_traj ---
    traj_embed_0: nn.Linear = model.stream_traj.embed[0]
    traj_embed_scores = _l1_neurons(traj_embed_0.weight.data, dim=0)
    keep_traj_embed = _topk_idx(traj_embed_scores, new_embed_dim)
    
    new_model.stream_traj.embed[0].weight.data.copy_(
        traj_embed_0.weight.data[keep_traj_embed]
    )
    new_model.stream_traj.embed[0].bias.data.copy_(
        traj_embed_0.bias.data[keep_traj_embed]
    )
    
    # stream_traj.temporal_encoder
    for i in [0, 1]:
        old_conv = model.stream_traj.temporal_encoder[i].conv
        new_conv = new_model.stream_traj.temporal_encoder[i].conv
        old_bn = model.stream_traj.temporal_encoder[i].bn
        new_bn = new_model.stream_traj.temporal_encoder[i].bn
        
        new_conv.weight.data.copy_(old_conv.weight.data[keep_traj_embed][:, keep_traj_embed])
        if old_conv.bias is not None:
            new_conv.bias.data.copy_(old_conv.bias.data[keep_traj_embed])
        
        new_bn.weight.data.copy_(old_bn.weight.data[keep_traj_embed])
        new_bn.bias.data.copy_(old_bn.bias.data[keep_traj_embed])
        new_bn.running_mean.data.copy_(old_bn.running_mean.data[keep_traj_embed])
        new_bn.running_var.data.copy_(old_bn.running_var.data[keep_traj_embed])
    
    # stream_traj.energy_gate: keep all (input is traj_channels=3, output is 1)
    new_model.stream_traj.energy_gate[0].weight.data.copy_(
        model.stream_traj.energy_gate[0].weight.data
    )
    new_model.stream_traj.energy_gate[0].bias.data.copy_(
        model.stream_traj.energy_gate[0].bias.data
    )
    
    # --- attention ---
    # attention[0]: Linear(fusion_dim, fusion_dim // 4)
    # attention[2]: Linear(fusion_dim // 4, fusion_dim)
    old_fusion_dim = old_embed_dim * 2
    new_fusion_dim = new_embed_dim * 2
    
    # Build combined keep indices for fusion (shape + traj)
    keep_fusion = torch.cat([keep_embed, keep_traj_embed + old_embed_dim])
    keep_fusion_new = torch.cat([
        torch.arange(new_embed_dim),
        torch.arange(new_embed_dim) + new_embed_dim
    ])
    
    old_attn_0 = model.attention[0]
    old_attn_2 = model.attention[2]
    
    # Compute scores for attention hidden
    attn_hidden_scores = _l1_neurons(old_attn_0.weight.data, dim=0)
    keep_attn_hidden = _topk_idx(attn_hidden_scores, new_fusion_dim // 4)
    
    new_model.attention[0].weight.data.copy_(
        old_attn_0.weight.data[keep_attn_hidden][:, keep_fusion]
    )
    new_model.attention[0].bias.data.copy_(
        old_attn_0.bias.data[keep_attn_hidden]
    )
    
    new_model.attention[2].weight.data.copy_(
        old_attn_2.weight.data[keep_fusion][:, keep_attn_hidden]
    )
    new_model.attention[2].bias.data.copy_(
        old_attn_2.bias.data[keep_fusion]
    )
    
    # --- classifier ---
    # classifier[0]: Linear(fusion_dim, embed_dim)
    # classifier[1]: LayerNorm(embed_dim)
    # classifier[4]: Linear(embed_dim, num_classes)
    old_cls_0 = model.classifier[0]
    old_cls_1 = model.classifier[1]
    old_cls_4 = model.classifier[4]
    
    cls_hidden_scores = _l1_neurons(old_cls_0.weight.data, dim=0)
    keep_cls_hidden = _topk_idx(cls_hidden_scores, new_embed_dim)
    
    new_model.classifier[0].weight.data.copy_(
        old_cls_0.weight.data[keep_cls_hidden][:, keep_fusion]
    )
    new_model.classifier[0].bias.data.copy_(
        old_cls_0.bias.data[keep_cls_hidden]
    )
    
    new_model.classifier[1].weight.data.copy_(
        old_cls_1.weight.data[keep_cls_hidden]
    )
    new_model.classifier[1].bias.data.copy_(
        old_cls_1.bias.data[keep_cls_hidden]
    )
    
    new_model.classifier[4].weight.data.copy_(
        old_cls_4.weight.data[:, keep_cls_hidden]
    )
    new_model.classifier[4].bias.data.copy_(
        old_cls_4.bias.data
    )
    
    meta = {
        "structured_spec": spec.__dict__,
        "old_embed_dim": old_embed_dim,
        "new_embed_dim": new_embed_dim,
        "kept_indices": {
            "shape_hidden": keep_hidden.tolist(),
            "shape_embed": keep_embed.tolist(),
            "traj_embed": keep_traj_embed.tolist(),
            "attn_hidden": keep_attn_hidden.tolist(),
            "cls_hidden": keep_cls_hidden.tolist(),
        },
    }
    
    return new_model, meta


def main():
    ap = argparse.ArgumentParser(description="Structured pruning for DSLNet (build smaller dense model)")
    ap.add_argument("--input", type=str, default="./checkpoints/dslnet_wlasl_100_best.pth")
    ap.add_argument("--output", type=str, default="./checkpoints/dslnet_wlasl_100_best_struct_pruned.pth")
    ap.add_argument("--num_classes", type=int, default=100)
    ap.add_argument("--num_joints", type=int, default=21)
    ap.add_argument("--embed_dim", type=int, default=48, help="Target embed_dim (default 64 -> 48)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    # Create model with original architecture
    model = DSLNet(num_classes=args.num_classes, config={'model': {'embed_dim': 64}}).eval()
    model.load_state_dict(state, strict=False)

    spec = StructuredSpec(embed_dim=args.embed_dim)
    new_model, meta = structured_prune_dslnet(model, spec, num_joints=args.num_joints)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": new_model.state_dict(),
        "structured_pruning": meta,
        "config": {'model': {'embed_dim': args.embed_dim}}
    }, args.output)

    in_mb = os.path.getsize(args.input) / 1024 / 1024
    out_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"Saved: {args.output}")
    print(f"Size: {in_mb:.2f} MB -> {out_mb:.2f} MB ({in_mb/out_mb:.2f}x smaller)")
    print(f"Spec: {spec}")


if __name__ == "__main__":
    main()

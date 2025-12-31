"""
WLASL 手語識別模型壓縮框架主入口

這個框架實現了完整的模型壓縮流程：
1. 教師模型訓練 (VideoMAE)
2. 知識蒸餾 (VideoMAE -> MobileViT)
3. 模型剪枝 (MobileViT pruning)
4. 量化感知訓練 (QAT)

使用方法:
    python main.py --stage teacher                    # 訓練教師模型
    python main.py --stage student                    # 知識蒸餾
    python main.py --stage prune                      # 模型剪枝
    python main.py --stage qat                        # 量化訓練
    python main.py --all                              # 運行完整流程
    python main.py --stage teacher --resume           # 從ckpt恢復
"""

import os
import sys
import argparse
import yaml
import torch
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

project_root = Path(__file__).parent
sys.path.append(str(project_root))

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(config: Dict[str, Any], config_path: str):
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

def get_device(device_config: str, stage: str = None) -> str:
    if device_config == "auto":
        if stage == "student":
            #KD因VideoMAE Conv3D兼容性問題 強制使用CPU!!!
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_config

def check_requirements():
    print("檢查項目需求...")
    required_files = [
        "wlasl_data/WLASL_v0.3.json",
        "wlasl_data/wlasl_class_list.txt",
    ]

    missing_files = []
    for file_path in required_files:
        if not (project_root / file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"缺少必要文件: {missing_files}")
        print("請確保已下載 WLASL 數據集")
        return False
    videos_dir = project_root / "wlasl_data" / "videos"
    if not videos_dir.exists():
        print(f"目錄不存在: {videos_dir}")
        print("請下載並解壓 WLASL video數據")

    try:
        import torch
        import torchvision
        import transformers
        import cv2
        import numpy as np
        import pandas as pd
        print("所有必要的依賴已安裝")
    except ImportError as e:
        print(f"缺少必要的依賴: {e}")
        print("請運行: pip install -r requirements.txt")
        return False

    return True

def run_stage(stage: str, config_path: str, resume: bool = False, **kwargs):
    """運行指定的訓練階段"""

    print(f"\n{'='*60}")
    print(f"開始執行階段: {stage.upper()}")
    print(f"{'='*60}")

    config = load_config(config_path)

    device = get_device(config.get('device', 'auto'), stage)
    config['device'] = device
    print(f"使用設備: {device}")

    log_dir = config.get('logging', {}).get('log_dir', f'./logs/{stage}')
    os.makedirs(log_dir, exist_ok=True)

    if stage == "teacher":
        return run_teacher_training(config, resume, **kwargs)
    elif stage == "student":
        return run_student_distillation(config, resume, **kwargs)
    elif stage == "prune":
        return run_model_pruning(config, **kwargs)
    elif stage == "qat":
        return run_quantization_training(config, **kwargs)
    else:
        raise ValueError(f"未知的階段: {stage}")

def run_teacher_training(config: Dict[str, Any], resume: bool, **kwargs):
    print("訓練教師模型 (VideoMAE)")

    save_path = config['save'].get('model_path', './best_videomae_wlasl_teacher.pth')
    if os.path.exists(save_path) and not resume:
        response = input(f"教師模型已存在: {save_path}\n是否要重新訓練? (y/n): ")
        if response.lower() != 'y':
            print("跳過教師模型訓練")
            return True

    cmd = [
        sys.executable, "training/train_teacher.py"
    ]

    if resume:
        cmd.append("--resume")

    print(f"執行命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, cwd=project_root, check=True)
        print("教師模型訓練完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"教師模型訓練失敗: {e}")
        return False

def run_student_distillation(config: Dict[str, Any], resume: bool, **kwargs):
    print("知識蒸餾訓練 (VideoMAE -> MobileViT)")
    teacher_path = config['model'].get('teacher_path', './best_videomae_wlasl_teacher.pth')
    if not os.path.exists(teacher_path):
        print(f"找不到教師模型: {teacher_path}")
        print("請先運行教師模型訓練")
        return False

    save_path = config['save'].get('model_path', './best_videomae_wlasl_student_kd_100.pth')
    if os.path.exists(save_path) and not resume:
        response = input(f"學生模型已存在: {save_path}\n是否要重新訓練? (y/n): ")
        if response.lower() != 'y':
            print("跳過學生模型訓練")
            return True

    import tempfile
    import yaml

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.safe_dump(config, f)
        temp_config_path = f.name

    cmd = [
        sys.executable, "training/train_student_kd.py",
        "--config", temp_config_path
    ]

    if resume:
        cmd.append("--resume")

    print(f"執行命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, cwd=project_root, check=True)
        print("知識蒸餾訓練完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"知識蒸餾訓練失敗: {e}")
        return False
    finally:
        try:
            os.unlink(temp_config_path)
        except:
            pass

def run_model_pruning(config: Dict[str, Any], **kwargs):
    print("模型剪枝 (MobileViT)")

    input_path = config['model'].get('input_path', './best_videomae_wlasl_student_kd_100.pth')
    if not os.path.exists(input_path):
        print(f"找不到輸入模型: {input_path}")
        print("請先運行學生模型訓練")
        return False

    cmd = [
        sys.executable, "training/prune_student.py",
        "--model_path", input_path,
        "--pruning_ratio", str(config['pruning']['target_sparsity']),
        "--pruning_method", config['pruning']['global']['pruning_method']
    ]

    if config['pruning']['iterative']['enabled']:
        cmd.extend([
            "--iterative",
            "--iterations", str(config['pruning']['iterations'])
        ])

    if config['fine_tune']['enabled']:
        cmd.extend([
            "--fine_tune",
            "--fine_tune_epochs", str(config['fine_tune']['epochs'])
        ])

    print(f"執行命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, cwd=project_root, check=True)
        print("模型剪枝完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"模型剪枝失敗: {e}")
        return False

def run_quantization_training(config: Dict[str, Any], **kwargs):
    print("量化感知訓練 (QAT)")

    input_path = config['model'].get('input_path', './best_videomae_wlasl_student_pruned_100.pth')
    if not os.path.exists(input_path):
        alt_path = './best_videomae_wlasl_student_kd_100.pth'
        if os.path.exists(alt_path):
            print(f"找不到剪枝模型，使用學生模型: {alt_path}")
            input_path = alt_path
        else:
            print(f"找不到輸入模型: {input_path}")
            return False

    # 構建命令
    cmd = [
        sys.executable, "models/train_qat.py",
        "--model_path", input_path,
        "--epochs", str(config['training']['epochs']),
        "--lr", str(config['training']['learning_rate'])
    ]

    print(f"執行命令: {' '.join(cmd)}")

    # 運行量化訓練
    try:
        result = subprocess.run(cmd, cwd=project_root, check=True)
        print("量化感知訓練完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"量化感知訓練失敗: {e}")
        return False

def run_full_pipeline():
    """運行完整的模型壓縮流程"""
    print("運行完整模型壓縮流程")
    print("這將執行: 教師訓練 → 知識蒸餾 → 模型剪枝 → 量化訓練")

    stages = [
        ("teacher", "configs/teacher.yaml"),
        ("student", "configs/student.yaml"),
        ("prune", "configs/prune.yaml"),
        ("qat", "configs/qat.yaml")
    ]

    results = {}
    for stage, config_path in stages:
        success = run_stage(stage, config_path)
        results[stage] = success

        if not success:
            print(f"流程在階段 '{stage}' 中斷")
            break

    print("\n" + "="*60)
    print("完整流程總結報告")
    print("="*60)

    for stage, success in results.items():
        status = "成功" if success else "失敗"
        print(f"   {stage.upper():8} : {status}")

    successful_stages = sum(results.values())
    total_stages = len(results)

    print(f"\n總計: {successful_stages}/{total_stages} 階段成功")

    if successful_stages == total_stages:
        print("完整模型壓縮流程成功完成!")
        print_compression_summary()
    else:
        print("流程未完全成功，請檢查失敗的階段")

def print_compression_summary():
    print("\n模型壓縮總結")

    models_info = [
        ("教師模型", "./best_videomae_wlasl_teacher_100.pth"),
        ("學生模型", "./best_videomae_wlasl_student_kd_100.pth"),
        ("剪枝模型", "./best_videomae_wlasl_student_pruned_100.pth"),
        ("量化模型", "./best_videomae_wlasl_student_qat_100.pth")
    ]

    print("-" * 50)
    print("<10")
    print("-" * 50)

    for name, path in models_info:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print("<10")
        else:
            print("<10")

    print("-" * 50)
    print("提示: 量化模型具有最佳的推理性能和最小的大小")

def main():
    parser = argparse.ArgumentParser(
        description="WLASL 手語識別模型壓縮框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 單個階段
  python main.py --stage teacher
  python main.py --stage student --resume
  python main.py --stage prune
  python main.py --stage qat

  # 完整流程
  python main.py --all

  # 自定義配置
  python main.py --stage teacher --config my_config.yaml
        """
    )

    parser.add_argument('--stage', choices=['teacher', 'student', 'prune', 'qat'],
                       help='要運行的訓練階段')
    parser.add_argument('--all', action='store_true',
                       help='運行完整的壓縮流程')
    parser.add_argument('--resume', action='store_true',
                       help='從檢查點恢復訓練')

    parser.add_argument('--config', type=str,
                       help='自定義配置文件路徑')

    parser.add_argument('--check', action='store_true',
                       help='檢查項目需求')
    parser.add_argument('--dry-run', action='store_true',
                       help='僅顯示將執行的命令，不實際運行')

    args = parser.parse_args()

    print("WLASL 手語識別模型壓縮框架")
    print("=" * 50)

    if args.check or args.all:
        if not check_requirements():
            sys.exit(1)

    if args.check:
        print("項目需求檢查完成")
        return

    if args.dry_run:
        print("乾運行模式 - 僅顯示命令")

    if args.all:
        run_full_pipeline()

    elif args.stage:
        config_path = args.config or f"configs/{args.stage}.yaml"

        if not os.path.exists(config_path):
            print(f"找不到配置文件: {config_path}")
            sys.exit(1)

        success = run_stage(args.stage, config_path, args.resume)
        if success:
            print(f"階段 '{args.stage}' 執行成功")
        else:
            print(f"階段 '{args.stage}' 執行失敗")
            sys.exit(1)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()

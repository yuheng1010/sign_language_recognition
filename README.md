# WLASL 手語識別模型壓縮框架

一個完整的從 VideoMAE 到輕量型 MobileViT 的模型壓縮解決方案。

### 環境設置

```bash
# 安裝依賴
pip install -r requirements.txt

# 下載 WLASL 數據集
# 請從官方網站下載並解壓到 wlasl_data/ 
```

### 檢查環境

```bash
# 檢查項目需求
python main.py --check
```

### 運行完整壓縮流程

```bash
# 一鍵運行完整流程 (推薦)
python main.py --all
```

## 詳細使用方法

### 階段性執行

```bash
# 1. 訓練教師模型 (VideoMAE)
python main.py --stage teacher

# 2. 知識蒸餾 (VideoMAE → MobileViT)
python main.py --stage student

# 3. 模型剪枝 (MobileViT)
python main.py --stage prune

# 4. 量化訓練 (QAT)
python main.py --stage qat
```

### 自定義配置

```bash
# 使用自定義配置文件
python main.py --stage teacher --config my_config.yaml

# 從檢查點恢復訓練
python main.py --stage student --resume
```

## 項目架構

```
├── configs/                # YAML 配置文件
│   ├── teacher.yaml       # 教師模型配置 (VideoMAE)
│   ├── student.yaml       # 學生模型配置 (知識蒸餾)
│   ├── prune.yaml         # 剪枝配置
│   └── qat.yaml          # 量化配置 (INT8 QAT)
├── training/              # 訓練腳本
│   ├── train_teacher.py      # 教師模型訓練 (50 epochs)
│   ├── train_student_kd.py   # 知識蒸餾訓練 (30 epochs)
│   └── prune_student.py      # 模型剪枝訓練 (迭代剪枝)
├── models/                # 模型定義與工具
│   ├── videomae.py           # VideoMAE 教師模型
│   ├── student_vit.py        # MobileViT 學生模型
│   ├── distillation.py       # 知識蒸餾損失函數
│   ├── prune_student.py      # 剪枝工具類
│   └── train_qat.py          # QAT 訓練腳本
├── data_pro/              # 數據處理
│   ├── dataset.py            # WLASL 數據集類
│   └── sampler.py            # 平衡採樣器 (處理類別不平衡)
├── wlasl_data/           # 數據集 (需要下載)
│   ├── videos/              # video文件 
│   ├── nlst_100.json             
│   ├── WLASL_v0.3.json      # 註釋文件
│   └── wlasl_class_list.txt # 類別列表 
├── main.py               # 統一入口腳本
└── README.md            
```

<!-- ## 壓縮效果

| 模型 | 大小 | Top-1準確度 | Top-5準確度 | 壓縮比例 |
|------|------|-------------|-------------|----------|
| VideoMAE (教師) | | 基準性能 | 基準性能 |  |
| MobileViT (學生) |  | |  |  |
| +剪枝 (30%) | |  |  |  |
| +量化 (INT8) |  | |  |  | -->

## 配置說明

### 教師模型配置 (configs/teacher.yaml)
- VideoMAE-base 預訓練模型
- 50個訓練輪數
- 完整的評估指標

### 學生模型配置 (configs/student.yaml)
- MobileViT-small 架構
- 知識蒸餾參數 (T=4.0, α=0.5)
- 平衡採樣處理類別不平衡

### 剪枝配置 (configs/prune.yaml)
- 迭代剪枝，目標稀疏度30%
- 自動微調保持性能
- 詳細的壓縮統計

### 量化配置 (configs/qat.yaml)
- INT8 量化
- QAT 訓練保持準確度
- 多種部署格式支持


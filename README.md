# ProgKGC: Progressive Structure-Enhanced Semantic Framework for Knowledge Graph Completion

This repository contains the **official implementation** of the paper:

> **ProgKGC: Progressive Structure-Enhanced Semantic Framework for Knowledge Graph Completion**  
> Zhuang Li, Yingwen Wu, Yachao Yuan, Jin Wang  
> Accepted at *ISWC 2025 (Research Track)*  

ProgKGC introduces a **progressive training strategy** and a **bidirectional neighbor aggregation mechanism** to effectively fuse semantic and structural signals in knowledge graph completion (KGC) tasks.

---

## 🔧 Features

- 🧠 **Progressive training**: gradually integrates structure after semantic learning
- 🔁 **Bidirectional aggregation**: incorporates both head and tail neighborhoods
- 📊 Supports evaluation on **FB15k-237** and **WN18RR**

---

## 📦 Installation

```bash
git clone https://github.com/0214ZhuangLi/ProgKGC.git
cd ProgKGC
pip install -r requirements.txt
```

## 🚀 Usage

### WN18RR

```bash
# Step 1: Preprocess
bash scripts/preprocess.sh WN18RR

# Step 2: Train
OUTPUT_DIR=./checkpoint/wn18rr/ bash scripts/train_wn.sh

# Step 3: Evaluate
bash scripts/eval.sh ./checkpoint/wn18rr/model_last.mdl WN18RR
```

### FB15k-237

```bash
# Step 1: Preprocess
bash scripts/preprocess.sh FB15k237

# Step 2: Train
OUTPUT_DIR=./checkpoint/fb15k237/ bash scripts/train_fb.sh

# Step 3: Evaluate
bash scripts/eval.sh ./checkpoint/fb15k237/model_last.mdl FB15k237
```

------

## 📁 Project Structure

```bash
ProgKGC/
├── data/                 # Datasets and preprocessed files
├── models/               # Model architecture
├── scripts/              # Training/evaluation scripts
├── utils/                # Utility functions
├── config/               # Hyperparameter configs
├── main.py               # Entry point
└── ...
```


# 🔍 Hallucination Detection in RAG Systems
### CS F429 — Natural Language Processing | BITS Pilani Dubai Campus
**Team 8:** Zayaan Ali · Aaqib Pangarkar · Ridhwan Ahamed · Mariah Shania

---

## 📌 Overview

This project tackles **hallucination detection** in Retrieval-Augmented Generation (RAG) systems — a critical open problem in NLP where language models generate fluent but factually incorrect or unsupported text.

We propose **Track A: Dynamic Uncertainty-Aware Attribution**, a training-free, token-level framework that identifies hallucinated spans in generated answers by measuring how a language model's probability distribution shifts when context is (and isn't) provided. Our method uses four uncertainty signals derived from two GPT-2 forward passes to produce a composite hallucination score per token.

> **No fine-tuning required. No labelled training data needed. Just two forward passes.**

---

## 🏗️ Method: Dynamic Uncertainty-Aware Attribution

The core insight: if a generated token is **grounded** in context, the model's confidence should *increase* when context is provided. If it is **hallucinated**, the context will have little to no effect.

We compute four token-level signals:

| Signal | Symbol | Description |
|---|---|---|
| **Information Gain** | ΔH | Entropy drop when context is added |
| **KL Divergence** | KL | Distribution shift between context-conditioned and unconditional passes |
| **Confidence Drop** | Δc | Change in max-token probability |
| **Semantic Entropy** | SE | Sequence-level average entropy (approximated per-token) |

These are combined into a **weighted composite score** (weights learned via logistic regression on RAGTruth):

```
composite = 0.15·IG + 0.55·KL + 0.30·ΔC
```

A token is labelled **HALLUCINATED** if its composite score falls below threshold τ = 0.50.

---

## 📊 Key Results

### Composite vs. Baselines (RAGTruth — AUROC)

| Method | AUROC |
|---|---|
| Entropy Only | 0.5338 |
| SelfCheckGPT | 0.6805 |
| **Our Composite** | **0.7243** |
| ReDeEP *(SOTA)* | 0.82 |
| LUMINA *(SOTA)* | 0.87 |

### Signal Ablation

| Signal | RAGTruth AUROC | HaluEval AUROC |
|---|---|---|
| Information Gain alone | 0.3482 | 0.4908 |
| KL Divergence alone | 0.6974 | 0.9060 |
| **Full Composite** | **0.7243** | **0.9249** |

### Hallucination Type Breakdown

| Hallucination Type | Count | Composite AUROC | Best Signal |
|---|---|---|---|
| Contradictory | 116 | **0.7841** | KL Divergence |
| Unsupported | 134 | 0.6902 | KL Divergence |

### Performance by Source LLM

| Source Model | AUROC |
|---|---|
| Llama-2-70B-chat | **0.8732** |
| GPT-4-0613 | 0.7877 |
| Llama-2-13B-chat | 0.7794 |
| Mistral-7B-Instruct | 0.7679 |
| GPT-3.5-Turbo | 0.6465 |
| Llama-2-7B-chat | 0.5346 |

---

## 📂 Repository Structure

```
NLP_Assignment/
│
├── 📓 Notebooks (run in order)
│   ├── 01_data_exploration.ipynb      # Dataset loading, EDA, label distribution analysis
│   ├── 02_core_pipeline.ipynb         # Core feature engineering & uncertainty signal computation
│   ├── 03_baselines.ipynb             # Baseline models (Entropy Only, SelfCheckGPT)
│   ├── 04_experiments.ipynb           # All 8 experiments: ablations, breakdowns, SOTA comparison
│   └── 05_demo .ipynb                 # Interactive demo on custom inputs
│
├── 🐍 Scripts
│   └── demo.py                        # CLI demo — run Track A on any passage/context pair
│
├── 📁 data/
│   ├── halueval.csv                   # HaluEval benchmark (hallucination evaluation)
│   ├── ragtruth_test_subset.csv       # RAGTruth test subset (250 samples)
│   └── ragtruth_test_subset_750.csv   # RAGTruth test subset (750 samples)
│   └── ...                            # (ragtruth.csv excluded — see Data Notes below)
│
├── 📁 results/
│   ├── experiment4_results.csv        # Signal ablation results
│   ├── experiment5_results.csv        # Hallucination-type breakdown
│   ├── experiment6_results.csv        # Per-source-model AUROC
│   ├── experiment8_sota.csv           # SOTA comparison table
│   └── ...                            # Merged result files & train metrics
│
├── 📁 outputs/
│   ├── experiment3_temporal.png       # Temporal analysis plot
│   └── experiment3_temporal_real.png  # Real temporal analysis plot
│
├── 📄 Deliverables
│   ├── NLP_Project_Report (1).pdf     # Final project report
│   ├── NLP_Report_Team_8.docx         # Report (Word format)
│   ├── NLP_PPT_TEAM_8.pptx            # Final presentation slides
│   ├── Assignment_NLP_CS_F429.pdf     # Original assignment specification
│   └── Midsem_Assignment_NLP_CS_F429 (1).pdf  # Mid-semester assignment
│
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/aaqibnp971/NLP_Assignment.git
cd NLP_Assignment
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv nlp_env
# Windows
nlp_env\Scripts\activate
# macOS/Linux
source nlp_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install torch transformers numpy scipy scikit-learn
pip install datasets pandas jupyterlab matplotlib seaborn
```

### 4. Run the CLI Demo

Test the hallucination detector on any passage/context pair:

```bash
python demo.py \
    --passage "Anne Frank was born in Frankfurt in 1929 and died in 1945." \
    --context "Anne Frank was a Jewish diarist born in Frankfurt am Main on 12 June 1929."
```

**Example output:**
```
TOKEN                       IG         KL  CONF_DROP       SE  COMPOSITE        LABEL
──────────────────────────────────────────────────────────────────────────────────────
' Anne'               0.1234     0.4521     0.3012   0.8734     0.3891     HALLUCINATED
' Frank'              0.9821     0.9234     0.8921   0.8734     0.9201         faithful
...
Sequence-level composite score : 0.6123
Sequence-level prediction      : FAITHFUL  (threshold τ = 0.50)
```

### 5. Run the Notebooks

```bash
jupyter lab
```

Open the notebooks in order: `01` → `02` → `03` → `04` → `05`.

---

## 📡 Datasets

| Dataset | Description | Source |
|---|---|---|
| **RAGTruth** | ~18K passage-level hallucination annotations from 6 LLMs across 3 RAG tasks | [Hugging Face — wandb/RAGTruth-processed](https://huggingface.co/datasets/wandb/RAGTruth-processed) |
| **HaluEval** | Question-answer hallucination pairs for cross-dataset generalisation | [HaluEval Benchmark](https://github.com/RUCAIBox/HaluEval) |

### ⚠️ Data Notes

- `ragtruth.csv` (~115 MB) is excluded from this repo due to GitHub's 100 MB file size limit.
- `01_data_exploration.ipynb` contains the code to pull and cache this dataset directly from the Hugging Face Hub.
- Smaller subsets (`ragtruth_test_subset.csv`, `ragtruth_test_subset_750.csv`) are included for quick experimentation.

---

## 🧪 Experiments

The `04_experiments.ipynb` notebook contains **8 numbered experiments**:

| # | Name | Description |
|---|---|---|
| 1 | Data Exploration | Label distribution, source model stats, task breakdown |
| 2 | Feature Engineering | Token-level signal computation on RAGTruth |
| 3 | Temporal Analysis | Hallucination position within answer sequences |
| 4 | Signal Ablation | Ablating each signal individually vs. full composite |
| 5 | Hallucination Type Breakdown | Contradictory vs. unsupported hallucinations |
| 6 | Per-Model Analysis | AUROC breakdown by source LLM |
| 7 | Cross-Dataset Generalisation | RAGTruth → HaluEval transfer |
| 8 | SOTA Comparison | Benchmarking against SelfCheckGPT, ReDeEP, LUMINA |

---

## 🛠️ Technical Details

- **Base Model**: GPT-2 (for uncertainty signal extraction — no fine-tuning)
- **Composite weights**: Learned by Logistic Regression on RAGTruth training split
  - Information Gain: **0.15**
  - KL Divergence: **0.55**
  - Confidence Drop: **0.30**
  - *(Semantic Entropy excluded — negative weight on training set)*
- **Decision threshold**: τ = **0.50** (tuned on validation set)
- **Evaluation metric**: AUROC (Area Under ROC Curve)
- **Hardware**: CPU-compatible (no GPU required for demo)

---

## 📋 Dependencies

| Package | Purpose |
|---|---|
| `torch` | GPT-2 inference |
| `transformers` | GPT-2 model & tokenizer |
| `numpy` / `scipy` | Signal computation |
| `scikit-learn` | Logistic Regression, AUROC evaluation |
| `datasets` | Hugging Face dataset loading |
| `pandas` | Data manipulation |
| `matplotlib` / `seaborn` | Visualisations |
| `jupyterlab` | Notebook environment |

---

## 👥 Team

| Name | Role |
|---|---|
| **Zayaan Ali** | Experiments, report writing |
| **Aaqib Pangarkar** | Core pipeline, demo implementation |
| **Ridhwan Ahamed** | Baselines, data analysis |
| **Mariah Shania** | Feature engineering, evaluation |

**Course**: CS F429 — Natural Language Processing
**Institution**: BITS Pilani, Dubai Campus

---

## 📚 References

- Zhang, Y., et al. *"RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models."* ACL 2024.
- Li, J., et al. *"HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models."* EMNLP 2023.
- Manakul, P., et al. *"SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models."* EMNLP 2023.
- Farquhar, S., et al. *"Detecting Hallucinations in Large Language Models Using Semantic Entropy."* Nature 2024.

---

<p align="center">
  <em>Built for CS F429 · BITS Pilani Dubai Campus · 2025</em>
</p>

# Exploring the Potential of LLMs in Healthcare
_A comparative evaluation of state-of-the-art language models across three medical tasks._

## 🎯 Project Overview
This project systematically evaluates the performance of **Large Language Models (LLMs)** on multiple healthcare-related datasets.  
The goal is to compare **accuracy, reasoning, and summarization quality** across leading open and proprietary models such as **ChatDoctor, GPT‑4o mini, Gemini, Grok‑3, and Llama‑3.1‑8B**.

## 🧠 Case Studies
| Case Study | Dataset | Evaluation Focus | Metrics |
|-------------|----------|------------------|----------|
| 🩺 Clinical Diagnosis | MedMCQA | Multiple-choice medical QA | **Accuracy** |
| 🧬 Biomedical Reasoning | PubMedQA | Yes/No/Maybe question answering | **Accuracy, Macro‑F1** |
| 📄 Discharge Summary | Asclepius Summarization | Medical text summarization | **ROUGE‑Lsum, BERTScore‑F1** |

## 📊 Poster Visuals
Three polished figures are generated for inclusion in research posters:

| Figure | File | Description |
|--------|------|--------------|
| MedMCQA Results | `viz_poster/medmcqa_poster.png` | Accuracy comparison across models |
| PubMedQA Results | `viz_poster/pubmedqa_poster.png` | Accuracy vs Macro‑F1 grouped bars |
| Asclepius Results | `viz_poster/asclepius_poster.png` | ROUGE‑Lsum vs BERTScore‑F1 grouped bars |

Each plot uses brand colours for clarity:  
- Deep Blue (`#001b47`) for primary metrics  
- Crimson Red (`#9e3638`) for secondary metrics

## ⚙️ Running the Code

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Generate evaluation figures
```bash
python poster_figs_final.py
```

### 3️⃣ Outputs
All visualizations and CSV summary tables are saved in:
```
viz_poster/
```

## 📦 Repository Structure
```
├── adapters/                     # Model interface wrappers (ChatDoctor, Gemini, Grok, etc.)
├── data/                         # Datasets (MedMCQA, PubMedQA, Asclepius)
├── outputs/                      # Raw model outputs + metric JSONs
├── viz_poster/                   # Final poster figures
│   ├── medmcqa_poster.png
│   ├── pubmedqa_poster.png
│   ├── asclepius_poster.png
│   └── *_poster_table.csv
├── llm_eval.py                   # Core evaluation runner
├── eval_summ_from_preds.py       # Summarization evaluation script
├── poster_figs_final.py          # Poster graph generator
└── README.md
```

## 🧩 Motivation
> To evaluate how well general-purpose and healthcare-tuned LLMs can handle diverse medical reasoning, summarization, and diagnosis tasks — improving reliability and clinical applicability of AI systems.

## 🧑‍💻 Author
**Subin Santhosh (a1917668)**  
Master of Computer Science, The University of Adelaide  
**Project:** Exploring the Potential of LLMs in Healthcare  
**Supervisor:** Prof. Hussain Ahmad

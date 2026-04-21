# Recursive Information-Theoretic Hierarchical Perception (Recursive-ITHP)

This repository hosts the **Recursive-ITHP** project, which fundamentally upgrades the Information-Theoretic Hierarchical Perception (ITHP) framework by introducing **Dynamic Recursive Perception**.

## 📖 The Story & Problem Solved

### The Bottleneck in Traditional Multimodal Fusion
Traditional multimodal learning models (such as those attempting sentiment analysis or sarcasm detection by combining Text, Audio, and Video) treat all modalities as equals. They fuse everything simultaneously using massive attention networks or feature concatenation. This causes two fatal issues:
1. **Information Redundancy & Noise:** Introducing raw visual and acoustic features alongside text often pollutes the latent space with irrelevant noise (e.g., background noise or unimportant facial twitches) rather than helping the prediction.
2. **Violation of Cognitive Physics:** The human brain does not process all senses equally at the exact same moment. It establishes a primary sensory pathway and pulls from other senses only to verify or supplement ambiguous information.

### The Original ITHP Solution
The original ITHP model solved this by proposing a **Hierarchical Perception** inspired by neuroscience. It assigned a "Prime Modality" (usually Text) and treated secondary modalities (Audio/Video) merely as "Detectors." Using the **Information Bottleneck (IB)** theory, it forcibly filtered out noise by maximizing the mutual information with the secondary modalities while strictly minimizing redundant mutual information from the primary input. 

### 🚀 Our Innovation: The "Recursive" Awakening
However, the original ITHP had a critical limitation: **its hierarchical depth was static and rigid.** 
Every sample, whether it was an obvious positive comment or a deeply masked sarcastic remark, went through the exact same processing depth. 

**Recursive-ITHP** solves this computational and cognitive rigidity by introducing **Dynamic Recursive Distillation**.
* **Dynamic Halting Mechanism:** We introduced a recursive semantic tree that acts dynamically. By utilizing a `halting_threshold`, the model evaluates the confidence of its latent state at each step. 
* **Adaptive Depth:** If the prime modality and a shallow fusion provide enough confidence for an obvious sample, the model **stops early**, saving computation and preventing overfitting. If a sentence is highly ambiguous (like complex sarcasm), the model recursively dives deeper, extracting secondary modality features multiple times (`max_recursion_depth`) until the Information Bottleneck is satisfied.
* **Syntax and Semantic Trees Integration:** Advanced syntax-aware losses and structural recursion deeply intertwine with the IB framework.

## 🏆 Performance Breakthroughs
By letting the model recursively decide its "thinking depth" (`avg_steps` usually hovering around 2.5 ~ 3.5), we achieved historic breakthroughs across major multimodal benchmarks:

- **CMU-MOSI**: Achieved an unprecedented **Test MAE of < 0.60** (Top 0.594) with **88.39% Binary Accuracy**.
- **CMU-MOSEI**: Reached an incredibly low **Test MAE of < 0.51** (Top 0.502) with **87.54% Accuracy**.
- **MUStARD & UR-FUNNY**: Extended to complex sarcasm and humor detection tasks using specialized HKT binary pipelines.

## 🛠️ Quick Start

### 1. Installation
Clone the repository and install dependencies tailored to the `ITHP5090` optimized environment:

```bash
git clone https://github.com/[your-repo]/Recursive-ITHP.git
cd Recursive-ITHP
pip install -r requirements.txt
```

### 2. Standard Training (MOSI / MOSEI)
Train the recursive model on standard CMU datasets:

```bash
python train.py --dataset mosi   
python train.py --dataset mosei 
```

### 3. Sarcasm & Humor Detection (MUStARD / UR-FUNNY)
We provide a dedicated HKT-style entry point for binary sarcasm/humor classification. This pipeline cleanly consumes HCF sequences and utilizes `BCEWithLogitsLoss`:

```bash
python train_hkt_binary.py --dataset mustard --train_batch_size 32
python train_hkt_binary.py --dataset urfunny --train_batch_size 32
```

### 4. Optuna Hyperparameter Search
The dynamic mechanics of recursion (e.g., `halting_threshold`, `syntax_temperature`, `max_recursion_depth`) are highly sensitive. We provide extensive `optuna` search scripts equipped with dual-phase TPE discrete searches. 
Optuna supports `load_if_exists=True`, allowing seamless background resume capabilities via SQLite.

```bash
# Global broad search
python scripts/optuna_search.py --dataset mosei --gpu 0 --output_dir optuna_results_mae

# Local refined search around the best anchor config
python scripts/optuna_local_refine_search.py --dataset mosi --gpu 1 --output_dir optuna_results_local_refine
```

## 🧬 Framework Details

- **`Recursive_ITHP.py` / `ITHP.py`**: The core models containing the Recursive Information Bottleneck logic.
- **`optuna_*.py`**: Scripts responsible for aggressively tuning the recursive depths and loss weights.
- **`train_hkt_binary.py`**: Specialized trainer adapting the recursive logic to binary tasks on heavily skewed datasets.

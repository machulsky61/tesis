# AI Safety via Debate - MNIST Experiment

This repository contains an implementation of the "AI Safety via Debate" experiment (Irving et al., 2018), applied to the MNIST dataset. The goal is to demonstrate how two agents compete to convince a "weaker" classifier (the judge) by selectively revealing a limited number of pixels, dramatically improving the classifier's accuracy.

---

## 📁 1. Folder Structure

```
.
├── agents/
│   ├── base_agent.py           # Base class for DebateAgent
│   ├── greedy_agent.py         # Greedy agent selecting pixels myopically
│   ├── greedy_adversarial_agent.py # Adversarial agent for judge evaluation
│   └── mcts_fast.py            # Fast MCTS implementation (replaces mcts_agent.py)
├── models/
│   ├── sparse_cnn.py           # SparseCNN model definition (judge, 2 channels)
│   ├── 28.pth                  # Pre-trained judge models
│   ├── 16.pth
│   ├── 28_4px.pth
│   └── 16_4px.pth
├── outputs/
│   ├── visualizations/         # Organized visualization outputs
│   │   ├── figures/            # Analysis graphs and plots
│   │   └── debates/            # Individual debate visualizations
│   ├── debates.csv             # Log of symmetric debates
│   ├── debates_asimetricos.csv # Log of asymmetric debates
│   ├── judges.csv              # Data on trained judge models
│   └── evaluations.csv         # Judge evaluation results
├── utils/
│   ├── data_utils.py           # Data loading and DebateDataset class
│   └── helpers.py              # Utility functions with colored visualization support
├── scripts/                    # Development and analysis tools
│   ├── analyze_results.py      # Statistical analysis and graph generation
│   ├── quick_examples.py       # Non-interactive examples
│   └── backups/                # Backup files
├── docs/                       # Documentation
│   └── example_usage.md        # Detailed usage guide
├── train_judge.py              # Train the judge with partial masks
├── eval_judge.py               # Enhanced judge evaluation with 7 strategies
├── run_debate.py               # Run debates between two agents (enhanced)
├── run_experiments.py          # Complete automation script
├── requirements.txt
└── README.md                   # This file
```

---

## 🔧 2. Installing Dependencies

```bash
python -m venv venv
source venv/bin/activate     # Linux/Mac
# .\venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

## 🚀 3. Quick Start

### 3.1 🤖 Complete Automation (Recommended)

```bash
python run_experiments.py
```
This interactive script handles everything: training judges, running experiments, and generating visualizations.

### 3.2 ⚡ Quick Examples (Non-interactive)

```bash
python scripts/quick_examples.py
```
Choose from pre-configured examples like high-precision experiments, agent comparisons, etc.

### 3.3 🎯 High-Precision Single Experiment

```bash
# 1 image, 2000 rollouts with colored visualization
python run_debate.py --judge_name 28 --agent_type mcts --rollouts 2000 --n_images 1 --save_colored_debate --note "high_precision"
```

### 3.4 🔄 Mixed Agents (MCTS vs Greedy)

```bash
# MCTS honest vs Greedy liar
python run_debate.py --judge_name 28 --mixed_agents --honest_agent mcts --rollouts 500 --n_images 50 --save_colored_debate --note "mixed_debate"
```

### 3.5 📊 Judge Evaluation (New Feature)

```bash
# Evaluate judge with different strategies
python eval_judge.py --judge_name 28 --strategy random --n_images 1000
python eval_judge.py --judge_name 28 --strategy optimal --n_images 1000
python eval_judge.py --judge_name 28 --strategy greedy_adversarial_agent --n_images 500
```

### 3.6 📈 Data Analysis

```bash
# Generate analysis graphs for thesis
python scripts/analyze_results.py --output-dir outputs/figures
```
---

## 📝 4. Replicating the Original Paper's Experiment

To replicate the original setup used in the paper, follow these steps:

### 4.1 Train the judge

```bash
python train_judge.py --judge_name og_judge
```

### 4.2 Run debates with both agents using 10k images

**Using MCTS (original configuration):**

```bash
python run_debate.py --judge_name og_judge --agent_type mcts --rollouts 10000 --n_images 10000
```

**Using Greedy (alternative faster but weaker agent):**

```bash
python run_debate.py --judge_name og_judge --agent_type greedy --n_images 10000
```

---

## 🌎 5. Key Flags and Arguments

Some arguments are used both when training the judge and running the debate, and **it's important to keep them consistent** between both stages for optimal performance.

* **Used in both training and debate:** `--resolution`, `--thr`, `--seed`, `--judge_name`, `--k`
* **Training only:** `--epochs`, `--batch_size`, `--lr`
* **Debate only:** `--agent_type`, `--rollouts`, `--n_images`, `--precommit`, `--starts`, `--save_*`

| Argument          | Description                                           | Values        | Default       |
| ----------------- | ----------------------------------------------------- | ------------- | ------------- |
| `--resolution`    | Input image size                                      | 16 or 28      | 28            |
| `--thr`           | Threshold for pixel intensity (judge pre-processing)  | 0.0 to 1.0    | 0.0           |
| `--seed`          | Random seed for reproducibility                       | int           | 42            |
| `--epochs`        | Number of training epochs for judge                   | int           | 64            |
| `--batch_size`    | Batch size for judge training                         | int           | 128           |
| `--lr`            | Learning rate                                         | float         | 1e-4          |
| `--judge_name`    | Identifier for saving/loading the judge               | str           | "judge\_name" |
| `--agent_type`    | Type of debating agent                                | greedy / mcts | greedy        |
| `--rollouts`      | Number of MCTS simulations (only used with mcts)      | int           | 50            |
| `--k`             | Number of pixels revealed during debate               | int           | 6             |
| `--n_images`      | Number of test images to run debates on               | int           | 100           |
| `--precommit`     | Liar agent precommits to a class label at the start   | store\_true   | False         |
| `--starts`        | Which agent starts first                              | honest / liar | liar          |
| `--save_images`   | Save PNG original images                              | store\_true   | False         |
| `--save_mask`     | Save PNG visualizations of masked images after debate | store\_true   | False         |
| `--save_play`     | Save JSON logs of pixel play sequences                | store\_true   | False         |
| `--save_metadata` | Save all metadata (images, mask, and play logs)       | store\_true   | False         |
| `--save_colored_debate` | Save colored visualization with move order      | store\_true   | False         |
| `--mixed_agents`  | Enable asymmetric debates (MCTS vs Greedy)            | store\_true   | False         |
| `--honest_agent`  | Type of honest agent when using mixed agents          | greedy / mcts | greedy        |
| `--allow_all_pixels` | Allow agents to select any pixel (not just >thr)    | store\_true   | False         |
| `--track_confidence` | Track judge confidence progression during debate     | store\_true   | False         |
| `--strategy`      | Pixel selection strategy for judge evaluation         | see eval_judge.py | random        |

---

## 📄 6. Module Descriptions

* **models/sparse\_cnn.py**: Defines `SparseCNN`, which classifies MNIST digits using masked images.

* **agents/**

  * `base_agent.py`: Provides shared functionality (image/judge handling).
  * `greedy_agent.py`: Picks the pixel maximizing immediate logit difference.
  * `greedy_adversarial_agent.py`: Adversarial agent that minimizes true class logits.
  * `mcts_fast.py`: Optimized MCTS with GPU-batched simulations.

* **utils/**

  * `data_utils.py`: Handles MNIST loading and DebateDataset creation.
  * `helpers.py`: General utilities for seeding, image/JSON saving, and CSV logging.

* **Main Scripts:**

  * `train_judge.py`: Trains judge model with masked data and saves it.
  * `eval_judge.py`: Evaluates judge with 7 different pixel selection strategies.
  * `run_debate.py`: Runs debates, logs outcomes, and evaluates accuracy.
  * `run_experiments.py`: Interactive automation script for complete workflows.

* **Analysis & Tools (`scripts/`):**

  * `analyze_results.py`: Statistical analysis and thesis-quality graph generation.
  * `quick_examples.py`: Pre-configured examples for common use cases.

---

## 📊 7. Citation

This implementation is based on:

> **AI Safety via Debate** – Geoffrey Irving, Paul Christiano, Dario Amodei (2018). \[[arXiv:1805.00899](https://arxiv.org/abs/1805.00899)]

---

## ⚒️ 8. Notes

* The judge is trained using randomly masked pixels, improving sparse image classification.
* When `--precommit` is enabled, the judge determines the winner based on the agent’s committed class.
* The liar agent will precommit to a class only if the flag is passed.
* **Agent Performance**: Greedy agents are fast, MCTS agents are accurate but slower.
* **Evaluation Strategies**: 7 different strategies available including adversarial agents for robustness testing.


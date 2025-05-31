# AI Safety via Debate - MNIST Experiment

This repository contains an implementation of the "AI Safety via Debate" experiment (Irving et al., 2018), applied to the MNIST dataset. The goal is to demonstrate how two agents compete to convince a "weaker" classifier (the judge) by selectively revealing a limited number of pixels, dramatically improving the classifier's accuracy.

---

## 📁 1. Folder Structure

```
.
├── agents/
│   ├── base_agent.py           # Base class for DebateAgent
│   ├── greedy_agent.py         # Greedy agent selecting pixels myopically
│   └── mcts_agent.py           # MCTS agent using Monte Carlo Tree Search
├── models/
│   ├── sparse_cnn.py           # SparseCNN model definition (judge, 2 channels)
│   └── judge_model.pth         # Pre-trained judge model
├── outputs/
│   ├── debate_id/              # Directory storing metadata of individual debates
│   ├── debates.csv             # Log of debates (accuracy, etc.)
│   └── judges.csv              # Data on trained judge models
├── utils/
│   ├── data_utils.py           # Data loading and DebateDataset class
│   └── helpers.py              # Utility functions (seeding, saving images/JSON, CSV logging)
├── train_judge.py              # Train the judge with partial masks
├── eval_judge.py               # Evaluate the judge with randomly masked images
├── run_debate.py               # Run debates between two agents
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

## 🚀 3. Quick Usage

### 3.1 Train the judge (16×16 pixels, quick test run):

```bash
python train_judge.py --resolution 16 --epochs 8
```

### 3.2 Test debates (greedy) on a small subset of images:

```bash
python run_debate.py --resolution 16 --agent_type greedy --precommit --n_images 100 --save_metadata
```

### 3.3 Run larger debates (MCTS)

```bash
python run_debate.py --resolution 16 --precommit --agent_type mcts --rollouts 1000 --n_images 1000
```

> 💡 **Tip:** MCTS is computationally expensive. For quick testing or demonstration purposes, it's better to use `greedy` with 1000–2000 images. Use 10000 images with `mcts` only if replicating the original paper.

### 3.4 (Optional) Repeat experiment for 4 pixels:
```bash
python train_judge.py --resolution 16 --k 4 --epochs 8 --judge_name 4px
python run_debate.py --resolution 16 --k 4 --judge_name 4px --precommit --agent_type greedy --n_images 1000
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

---

## 📄 6. Module Descriptions

* **models/sparse\_cnn.py**: Defines `SparseCNN`, which classifies MNIST digits using masked images.

* **agents/**

  * `base_agent.py`: Provides shared functionality (image/judge handling).
  * `greedy_agent.py`: Picks the pixel maximizing immediate logit difference.
  * `mcts_agent.py`: Uses Monte Carlo simulations to select optimal pixels.

* **utils/**

  * `data_utils.py`: Handles MNIST loading and DebateDataset creation.
  * `helpers.py`: General utilities for seeding, image/JSON saving, and CSV logging.

* **scripts/**

  * `train_judge.py`: Trains judge model with masked data and saves it.
  * `eval_judge.py`: Evaluates judge model accuracy and logs results.
  * `run_debate.py`: Runs debates, logs outcomes, and evaluates accuracy.

---

## 📊 7. Citation

This implementation is based on:

> **AI Safety via Debate** – Geoffrey Irving, Paul Christiano, Dario Amodei (2018). \[[arXiv:1805.00899](https://arxiv.org/abs/1805.00899)]

---

## ⚒️ 8. Notes

* The judge is trained using randomly masked pixels, improving sparse image classification.
* When `--precommit` is enabled, the judge determines the winner based on the agent’s committed class.
* The liar agent will precommit to a class only if the flag is passed.
* Greedy agents are fast and useful for debugging or small-scale tests.
* MCTS agents are more accurate but significantly slower, making them ideal for high-confidence evaluations or paper replication scenarios.
* You can experiment with different values of `--k`, such as `--k 4` or `--k 6`, depending on how much information the agents are allowed to reveal.

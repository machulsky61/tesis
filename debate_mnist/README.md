# AI Safety via Debate · MNIST Experiment

This repository contains an implementation of the “AI Safety via Debate” experiment
(Irving et al., 2018) applied to the MNIST dataset. The goal is to show how two agents
compete to convince a “weaker” classifier (the judge) by revealing only a few
pixels, and how this dramatically improves its accuracy.

---

## 📁 Folder Structure

```
.
├── models/
│   └── sparse_cnn.py           # Definition of the SparseCNN model (judge, 2 channels)
├── agents/
│   ├── base_agent.py           # Base DebateAgent class
│   ├── greedy_agent.py         # Greedy agent (myopically selects pixel)
│   └── mcts_agent.py           # MCTS agent (Monte Carlo search)
├── utils/
│   ├── data_utils.py           # Data loading and DebateDataset
│   └── helpers.py              # Seeding, image/JSON saving, and CSV logging
├── scripts/
│   ├── train_judge.py          # Trains the judge with partial masks
│   └── run_debate.py           # Runs debates between two agents
└── README.md                   # This file
```

---

## 🔧 Installing Dependencies

```bash
python -m venv venv
source venv/bin/activate     # Linux/Mac
# .\venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

## 🚀 Quick Usage

1. **Train the judge** (16×16 pixels, few epochs):

```bash
python scripts/train_judge.py \
    --resolution 16 \
    --thr 0.1 \
    --seed 42 \
    --epochs 3 \
    --batch_size 128 \
    --lr 1e-3 \
```


2. **Test debates (greedy)** on a few images:

 ```bash
python scripts/run_debate.py \
    --resolution 16 \
    --thr 0.1 \
    --agent_type greedy \
    --k 6 \
    --seed 42 \
    --n_images 20 \
    --save_images \
    --save_metadata \
```

3. **Run a larger Debate (MCTS Agent)**:

### 3. 
```bash
python scripts/run_debate.py \
    --resolution 16 \
    --thr 0.1 \
    --agent_type mcts \
    --rollouts 3000 \
    --k 6 \
    --seed 42 \
    --n_images 1000 \
    --log_csv experiments_log.csv
```

4. **(Optional) Repeat with 28×28**:

```bash
python scripts/train_judge.py --epochs 64 --seed 42 --judge_name judge_28px
python scripts/run_debate.py --judge_name judge_28px --agent_type mcts --rollouts 10000 --k 6 --seed 42 --n_images 10000
```

---

## 🌎 Key Flags and Arguments

| Argument         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `--resolution`   | Input image size: 16 or 28 (default: 16)                                     |
| `--thr`          | Threshold for pixel intensity (used in judge pre-processing)                 |
| `--seed`         | Random seed                                                                  |
| `--epochs`       | Number of training epochs for the judge                                     |
| `--batch_size`   | Batch size during training                                                   |
| `--lr`           | Learning rate                                                                |
| `--judge_name`   | Identifier to save/load the judge model                                     |
| `--agent_type`   | `greedy` or `mcts`                                                           |
| `--rollouts`     | Number of MCTS simulations (only used if `agent_type=mcts`)                |
| `--k`            | Total number of pixels revealed in a debate                                 |
| `--n_images`     | Number of test images to run debates on                                     |
| `--precommit`    | If set, agents precommit to a class label at the start                      |
| `--starts`       | Which agent starts first (`honest` or `liar`)                               |
| `--save_images`  | Save PNG visualizations of debates                                          |
| `--save_metadata`| Save JSON logs of debates                                                   |
| `--log_csv`      | Path to CSV log where results will be appended    

---

## 📄 Module Descriptions

* **models/sparse\_cnn.py**
  Defines the `SparseCNN` network, which receives two channels (mask and values) and classifies MNIST digits.

* **agents/**

  * `base_agent.py`: Base class with shared functionality (image and judge handling).
  * `greedy_agent.py`: Selects the pixel that immediately maximizes the logit difference in favor of its class.
  * `mcts_agent.py`: Performs random simulations (Monte Carlo) to choose the pixel with the highest win rate.

* **utils**

  * `data\_utils.py`: Loads MNIST and builds a `DebateDataset` that, during training, generates partial masks with relevant pixels to teach the judge.
  * `helpers.py`: Utilities for seeding, saving images/JSON, and unified CSV logging for experiments.

* **scripts**
  * `train\_judge.py`: Trains the judge model with partial data (mask + values) and saves its state.
  * `eval\_judge.py`: Evaluates the judge model, reports accuracy and saves results.
  * `run\_debate.py`: Coordinates debates between two agents and the trained judge, saves results (CSV, images, JSON), and reports accuracy.

---

---

## 📊 Citation
This implementation is inspired by the original paper:
> **AI Safety via Debate** — Geoffrey Irving, Paul Christiano, Dario Amodei (2018). [[arXiv:1805.00899](https://arxiv.org/abs/1805.00899)]

---

## ⚒️ Notes
- The sparse judge is trained by randomly revealing a subset of pixels during training.
- When using `--precommit`, the winning condition is based on which class the judge picks between the two agents' claims.
- Agents use different strategies: `greedy` always picks the most favorable next pixel, while `mcts` performs simulations to select actions. Therefore it takes longer to run.

---

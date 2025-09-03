# AI Safety via Debate - MNIST Implementation

Implementation of "AI Safety via Debate" (Irving et al., 2018) on MNIST dataset. Two agents compete to convince a judge classifier by selectively revealing pixels.

## Quick Start

### Installation
```bash
cd debate_mnist
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Basic Usage

**1. Train a judge model:**
```bash
python train_judge.py --judge_name my_judge --resolution 28 --epochs 8
```

**2. Run debates:**
```bash
# Quick test with Greedy agents
python run_debate.py --judge_name my_judge --agent_type greedy --n_images 100

# Higher quality with MCTS agents
python run_debate.py --judge_name my_judge --agent_type mcts --rollouts 500 --n_images 100
```

**3. Complete automation:**
```bash
python run_experiments.py
```

## Key Features

- **Multiple Agent Types**: Greedy (fast) and MCTS (high-quality) debate agents
- **Judge Evaluation**: Test judge robustness with different pixel selection strategies
- **Mixed Agent Debates**: MCTS vs Greedy asymmetric competitions
- **Precommit Strategy**: Liar agent commits to wrong class at start
- **Comprehensive Logging**: CSV exports and JSON metadata for analysis
- **Logits Tracking**: Monitor judge decision-making throughout debates

## Important Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--judge_name` | Model identifier (must match between training/debate) | required |
| `--resolution` | Image size (16 or 28) | 28 |
| `--k` | Number of pixels revealed per debate | 6 |
| `--agent_type` | Agent type (greedy/mcts) | greedy |
| `--rollouts` | MCTS simulations (higher = better quality) | 50 |
| `--n_images` | Number of test images | 100 |
| `--precommit` | Enable precommit strategy | False |
| `--mixed_agents` | Enable MCTS vs Greedy debates | False |
| `--save_metadata` | Save images and visualizations | False |

## Outputs

- **debates.csv** - Debate results with accuracy metrics
- **evaluations.csv** - Judge performance with different strategies  
- **judges.csv** - Training metadata for judge models
- **Visualizations** - Colored debate progression images (if enabled)
- **JSON Metadata** - Detailed logs with logits tracking (if enabled)

## Paper Replication

Train judge and run 10k image experiments:
```bash
python train_judge.py --judge_name paper_judge
python run_debate.py --judge_name paper_judge --agent_type mcts --rollouts 10000 --n_images 10000
```

## Citation

Based on: **AI Safety via Debate** – Irving, Christiano, Amodei (2018). [arXiv:1805.00899](https://arxiv.org/abs/1805.00899)
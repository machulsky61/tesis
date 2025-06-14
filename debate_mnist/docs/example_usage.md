# 🧪 Usage Guide: Experiment Automation Script

## Running the Script

```bash
python run_experiments.py
```

## Interactive Session Example

### 1. Initial Banner
```
================================================================================
🧪 COMPLETE DEBATE EXPERIMENTS AUTOMATION SCRIPT 🧪
================================================================================
Complete script for:
• Training judge models
• Running debate experiments
• Generating configurable visualizations
• Results analysis
```

### 2. Phase 1: Judge Training

**Question:** `Train judge models? (y/n) [n]:`

If you answer **`y`**:
```
================================================================================
🎓 JUDGE MODEL TRAINING
================================================================================

Available judge configurations:
  28: Resolution 28x28, 6 pixels - ✅ Exists
  16: Resolution 16x16, 6 pixels - ❌ Does not exist
  28_4px: Resolution 28x28, 4 pixels - ❌ Does not exist
  16_4px: Resolution 16x16, 4 pixels - ❌ Does not exist

Which judge models do you want to train?

Train 16 (Resolution 16x16, 6 pixels) (y/n) [n]: y
Epochs for 16 [64]: 32

Train 28_4px (Resolution 28x28, 4 pixels) (y/n) [n]: y
Epochs for 28_4px [64]: 64

Train a judge with custom configuration? (y/n) [n]: n

📋 2 judge models will be trained:
  - 16: Resolution 16x16, 6 pixels (32 epochs)
  - 28_4px: Resolution 28x28, 4 pixels (64 epochs)

Proceed with training? (y/n) [n]: y
```

### 3. Phase 2: Visualization Configuration

**Question:** `Enable visualizations? (y/n) [n]:`

If you answer **`y`**:
```
================================================================================
🎨 VISUALIZATION CONFIGURATION
================================================================================

Visualization types:
• Save colored debate images? (--save_colored_debate) (y/n) [n]: y
• Save complete metadata? (--save_metadata) (y/n) [n]: y

📊 CUSTOM VISUALIZATION EXPERIMENTS
These are specific experiments to generate high-quality visualizations.

Create custom visualization experiments? (y/n) [n]: y

• High precision experiment (1 image, many rollouts) (y/n) [n]: y
Agent type (greedy/mcts/mixed) [mcts]: mcts
Number of rollouts [2000]: 3000

• Comparison experiment (same seed, different agents) (y/n) [n]: y
Fixed seed [42]: 123
Number of images [10]: 5
Rollouts for MCTS [1000]: 1500

• Rollout scalability experiment (same image, different rollouts) (y/n) [n]: y
Fixed seed [42]: 456
Select rollout values:
  - 50 rollouts (y/n) [n]: y
  - 100 rollouts (y/n) [n]: y
  - 500 rollouts (y/n) [n]: y
  - 1000 rollouts (y/n) [n]: y

📋 7 custom visualization experiments created.
```

### 4. Phase 3: Global Configuration

```
================================================================================
📋 GLOBAL EXPERIMENT CONFIGURATION
================================================================================

Available judge models: 28, 16, 28_4px
Judge model name (28/16/28_4px) [28]: 28_4px

Using judge '28_4px' with resolution 28x28

Number of images for Greedy experiments [1000]: 500
Number of images for MCTS experiments [500]: 300
Number of rollouts for MCTS [500]: 800
```

### 5. Experiment Selection

```
================================================================================
🎯 EXPERIMENT SELECTION
================================================================================

1️⃣ SYMMETRIC EXPERIMENTS (same agent type)

Run Greedy vs Greedy experiments? (y/n) [n]: y
  - Greedy baseline (no precommit, liar first) (y/n) [n]: y
  - Greedy with precommit (y/n) [n]: y
  - Greedy with honest first (y/n) [n]: n
  - Greedy with precommit + honest first (y/n) [n]: n

Run MCTS vs MCTS experiments? (y/n) [n]: y
  - MCTS baseline (y/n) [n]: y
  - MCTS with precommit (y/n) [n]: y

2️⃣ ASYMMETRIC EXPERIMENTS (MCTS vs Greedy)

Run MCTS Honest vs Greedy Liar experiments? (y/n) [n]: y
  - MCTS honest vs Greedy liar baseline (y/n) [n]: y
  - MCTS honest vs Greedy liar with precommit (y/n) [n]: y

Run Greedy Honest vs MCTS Liar experiments? (y/n) [n]: n

3️⃣ SPECIAL EXPERIMENTS

Run experiments with different MCTS rollout numbers? (y/n) [n]: y
  - MCTS with 100 rollouts (y/n) [n]: y
  - MCTS with 200 rollouts (y/n) [n]: y
  - MCTS with 1000 rollouts (y/n) [n]: n

4️⃣ VISUALIZATION EXPERIMENTS

Create visual samples of each agent type? (y/n) [n]: y
  - Greedy visualization (y/n) [n]: y
  - MCTS visualization (y/n) [n]: y
  - Mixed MCTS honest visualization (y/n) [n]: y
```

### 6. Final Summary

```
================================================================================
📊 SELECTED EXPERIMENTS SUMMARY
================================================================================
Total experiments: 15

Experiments to execute:
 1. Greedy vs Greedy - baseline
 2. Greedy vs Greedy - precommit
 3. MCTS vs MCTS - baseline
 4. MCTS vs MCTS - precommit
 5. MCTS Honest vs Greedy Liar - baseline
 6. MCTS Honest vs Greedy Liar - precommit
 7. MCTS with 100 rollouts
 8. MCTS with 200 rollouts
 9. Greedy visualization
10. MCTS visualization
11. Mixed MCTS honest visualization
12. High precision MCTS 3000 rollouts
13. Greedy comparison seed123
14. MCTS comparison seed123
15. MCTS scalability 50 rollouts

Proceed with execution of all experiments? (y/n) [n]: y
```

### 7. Execution

```
================================================================================
🚀 STARTING EXPERIMENT EXECUTION
================================================================================

🔄 Progress: 1/15 (6.7%)
============================================================
🚀 Executing: Greedy vs Greedy - baseline
Command: python run_debate.py --judge_name 28_4px --resolution 28 --agent_type greedy --n_images 500 --note greedy_baseline
============================================================
✅ Completed successfully

🔄 Progress: 2/15 (13.3%)
...
```

### 8. Final Results

```
================================================================================
🏁 EXPERIMENTS COMPLETED
================================================================================
✅ Successful: 15
❌ Failed: 0
📊 Results saved in:
   - outputs/debates.csv (symmetric debates)
   - outputs/debates_asimetricos.csv (asymmetric debates)
   - outputs/debate_*/ (visualizations)

🎉 Experiments completed! Check CSV files to analyze results.
```

## Specific Use Cases

### Train Judges Only
```bash
python run_experiments.py
# Answer 'y' only to training, 'n' to everything else
```

### High Precision Visualizations Only
```bash
python run_experiments.py
# 'n' to training
# 'y' to visualizations → 'y' to custom experiments → 'y' to high precision
# 'n' to normal experiments
```

### Quick Testing Experiments
```bash
python run_experiments.py
# Use small numbers of images (10-50)
# Only Greedy experiments (faster)
```

## Output Files

- **`outputs/debates.csv`**: Symmetric debates (same agent type)
- **`outputs/debates_asimetricos.csv`**: Mixed debates (MCTS vs Greedy)
- **`outputs/debate_*/`**: Folders with colored visualizations
- **`outputs/judges.csv`**: Judge training logs

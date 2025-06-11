#!/usr/bin/env python3
"""
Analysis script for debate results visualization.
Generates graphs for thesis sections 3.2, 3.4, and 3.5.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

# Set style and colors
plt.style.use('seaborn-v0_8')

# Color scheme definition
COLORS = {
    'honest': '#2E86AB',      # Blue for honest
    'liar': '#A23B72',        # Red for liar
    'greedy': '#62C370',      # Light green for greedy
    'mcts': '#1B4332',        # Dark green for MCTS
    'judge': '#8E44AD',       # Purple for judge
    'precommit': '#2E86AB',   # Blue for precommit
    'no_precommit': '#F18F01', # Orange for no precommit
    'neutral': '#34495E'      # Dark gray for neutral elements
}

# Set professional color palette
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', 
    ['#2E86AB', '#A23B72', '#62C370', '#1B4332', '#8E44AD', '#F18F01'])
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def load_data():
    """Load all CSV data files."""
    data_path = Path("outputs")
    
    # Load debates data
    debates_df = pd.read_csv(data_path / "debates.csv")
    
    # Load judge evaluations (random pixels baseline) - now properly formatted
    evaluations_df = pd.read_csv(data_path / "evaluations.csv")
    
    # Convert to proper types
    evaluations_df['pixels'] = pd.to_numeric(evaluations_df['pixels'], errors='coerce')
    evaluations_df['accuracy'] = pd.to_numeric(evaluations_df['accuracy'], errors='coerce')
    
    # Keep only rows with valid data
    evaluations_df = evaluations_df.dropna(subset=['pixels', 'accuracy'])
    
    # Load judge training metadata
    judges_df = pd.read_csv(data_path / "judges.csv")
    
    return debates_df, evaluations_df, judges_df

def create_graph1_judge_vs_debate(debates_df, evaluations_df, output_dir="outputs/figures"):
    """
    Graph 1: Judge Precision vs Debate (Section 3.2)
    Grouped bar chart comparing judge alone vs debate performance.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data for k=4 and k=6
    k_values = [4, 6]
    x_pos = np.arange(len(k_values))
    width = 0.25
    
    # Data storage
    judge_alone = []
    greedy_honest = []
    mcts_honest = []
    
    for k in k_values:
        # Judge baseline (random pixels)
        judge_data = evaluations_df[evaluations_df['pixels'] == k]
        if not judge_data.empty:
            judge_baseline = judge_data['accuracy'].mean()
            if pd.isna(judge_baseline):
                judge_baseline = 0
        else:
            judge_baseline = 0
        judge_alone.append(judge_baseline * 100)
        
        # Greedy debate - honest success rate with precommit
        greedy_data = debates_df[
            (debates_df['pixels'] == k) & 
            (debates_df['agent_type'] == 'greedy') & 
            (debates_df['precommit'] == True)
        ]
        greedy_honest_rate = greedy_data['accuracy'].mean()
        if pd.isna(greedy_honest_rate):
            greedy_honest_rate = 0
        greedy_honest.append(greedy_honest_rate * 100)
        
        # MCTS debate - honest success rate with precommit
        mcts_data = debates_df[
            (debates_df['pixels'] == k) & 
            (debates_df['agent_type'] == 'mcts') & 
            (debates_df['precommit'] == True)
        ]
        mcts_honest_rate = mcts_data['accuracy'].mean() if not mcts_data.empty else 0
        if pd.isna(mcts_honest_rate):
            mcts_honest_rate = 0
        mcts_honest.append(mcts_honest_rate * 100)
    
    # Calculate y-axis limits for better zoom
    all_values = [v for v in judge_alone + greedy_honest + mcts_honest if v > 0]
    if all_values:
        y_min = max(0, min(all_values) - 5)
        y_max = min(100, max(all_values) + 5)
    else:
        y_min, y_max = 0, 100
    
    # Create bars with custom colors
    bars1 = ax.bar(x_pos - width, judge_alone, width, 
                   label='Sparse CNN Judge (Random Pixels)', 
                   color=COLORS['judge'], alpha=0.9, edgecolor='white', linewidth=1)
    bars2 = ax.bar(x_pos, greedy_honest, width, 
                   label='Greedy Debate (Precommit)', 
                   color=COLORS['greedy'], alpha=0.9, edgecolor='white', linewidth=1)
    bars3 = ax.bar(x_pos + width, mcts_honest, width, 
                   label='MCTS Debate (Precommit)', 
                   color=COLORS['mcts'], alpha=0.9, edgecolor='white', linewidth=1)
    
    # Customize chart
    ax.set_xlabel('Number of Revealed Pixels (k)', fontweight='bold')
    ax.set_ylabel('Judge Accuracy after Debate (%)', fontweight='bold')
    ax.set_title('Judge Precision vs. Debate', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'k={k}' for k in k_values], fontsize=12)
    
    
    # Enhanced grid with fixed scale (steps of 5)
    y_min_rounded = int(y_min // 5) * 5
    y_max_rounded = int((y_max + 4) // 5) * 5
    y_ticks = np.arange(y_min_rounded, y_max_rounded + 1, 5)
    ax.set_yticks(y_ticks)
    ax.set_ylim(y_min_rounded, y_max_rounded)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Enhanced legend
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}%', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/graph1_judge_vs_debate.png", dpi=300, bbox_inches='tight')
    plt.show()

def create_graph2_precommit_impact(debates_df, output_dir="outputs/figures"):
    """
    Graph 2: Precommit Impact (Section 3.4)
    Grouped bar chart showing with/without precommit comparison.
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Get unique combinations
    conditions = []
    with_precommit = []
    without_precommit = []
    bar_colors_with = []
    bar_colors_without = []
    
    # Define conditions to analyze
    analysis_conditions = [
        ('greedy', 4, 'honest'),
        ('greedy', 4, 'liar'),
        ('greedy', 6, 'honest'),
        ('greedy', 6, 'liar'),
        ('mcts', 4, 'honest'),
        ('mcts', 4, 'liar'),
        ('mcts', 6, 'honest'),
        ('mcts', 6, 'liar')
    ]
    
    for agent_type, k, starter in analysis_conditions:
        condition_data = debates_df[
            (debates_df['agent_type'] == agent_type) & 
            (debates_df['pixels'] == k) & 
            (debates_df['started'] == starter)
        ]
        
        if condition_data.empty:
            continue
            
        # With precommit
        with_data = condition_data[condition_data['precommit'] == True]
        with_rate = with_data['accuracy'].mean() * 100 if not with_data.empty else 0
        
        # Without precommit
        without_data = condition_data[condition_data['precommit'] == False]
        without_rate = without_data['accuracy'].mean() * 100 if not without_data.empty else 0
        
        # Only add if we have data for both conditions
        if with_rate > 0 and without_rate > 0:
            condition_label = f"{agent_type.title()}, {k}px\n{starter.title()} Starts"
            conditions.append(condition_label)
            with_precommit.append(with_rate)
            without_precommit.append(without_rate)
            
            # Determine colors based on starter
            if starter == 'honest':
                bar_colors_with.append(COLORS['honest'])
                bar_colors_without.append(COLORS['honest'])
            else:
                bar_colors_with.append(COLORS['liar'])
                bar_colors_without.append(COLORS['liar'])
    
    if not conditions:
        print("No data available for precommit comparison")
        return
    
    # Calculate y-axis limits for better zoom
    all_values = with_precommit + without_precommit
    if all_values:
        y_min = max(0, min(all_values) - 5)
        y_max = min(100, max(all_values) + 5)
    else:
        y_min, y_max = 0, 100
    
    # Create bars
    x_pos = np.arange(len(conditions))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, with_precommit, width, 
                   label='Precommitment', color=COLORS['precommit'], 
                   alpha=0.9, edgecolor='white', linewidth=1)
    bars2 = ax.bar(x_pos + width/2, without_precommit, width, 
                   label='No Precommitment', color=COLORS['no_precommit'], 
                   alpha=0.9, edgecolor='white', linewidth=1)
    
    # Customize chart
    ax.set_xlabel('Conditions (Agent, Pixels, Starter)', fontweight='bold')
    ax.set_ylabel('Judge Accuracy after Debate (%)', fontweight='bold')
    ax.set_title('Precommitment Impact', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions, rotation=0, ha='center', fontsize=11)
    
    # Set y-axis limits for better zoom
    ax.set_ylim(y_min, y_max)
    
    # Enhanced grid with fixed scale (steps of 5)
    y_min_rounded = int(y_min // 5) * 5
    y_max_rounded = int((y_max + 4) // 5) * 5
    y_ticks = np.arange(y_min_rounded, y_max_rounded + 1, 5)
    ax.set_yticks(y_ticks)
    ax.set_ylim(y_min_rounded, y_max_rounded)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Enhanced legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Add value labels
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}%', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/graph2_precommit_impact.png", dpi=300, bbox_inches='tight')
    plt.show()

def create_graph3_agent_comparison(debates_df, output_dir="outputs/figures"):
    """
    Graph 3: Agent Type Comparison (Section 3.5)
    Grouped bar chart comparing Greedy vs MCTS agents.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define conditions for comparison
    conditions = []
    greedy_rates = []
    mcts_rates = []
    
    # Analysis conditions (focusing on k=6 where we have MCTS data)
    comparison_conditions = [
        (6, True, 'honest'),
        (6, True, 'liar'),
        (6, False, 'honest'),
        (6, False, 'liar')
    ]
    
    for k, precommit, starter in comparison_conditions:
        # Greedy performance
        greedy_data = debates_df[
            (debates_df['agent_type'] == 'greedy') & 
            (debates_df['pixels'] == k) & 
            (debates_df['precommit'] == precommit) & 
            (debates_df['started'] == starter)
        ]
        greedy_rate = greedy_data['accuracy'].mean() * 100 if not greedy_data.empty else 0
        
        # MCTS performance
        mcts_data = debates_df[
            (debates_df['agent_type'] == 'mcts') & 
            (debates_df['pixels'] == k) & 
            (debates_df['precommit'] == precommit) & 
            (debates_df['started'] == starter)
        ]
        mcts_rate = mcts_data['accuracy'].mean() * 100 if not mcts_data.empty else 0
        
        # Only add if we have data for at least one agent type
        if greedy_rate > 0 or mcts_rate > 0:
            precommit_str = "" if precommit else "No "
            condition_label = f"{k}px, {precommit_str}Precommit\nStarted: {starter.title()}"
            conditions.append(condition_label)
            greedy_rates.append(greedy_rate)
            mcts_rates.append(mcts_rate)
    
    if not conditions:
        print("No data available for agent comparison")
        return
    
    # Calculate y-axis limits for better zoom
    all_values = [v for v in greedy_rates + mcts_rates if v > 0]
    if all_values:
        y_min = max(0, min(all_values) - 5)
        y_max = min(100, max(all_values) + 5)
    else:
        y_min, y_max = 0, 100
    
    # Create bars
    x_pos = np.arange(len(conditions))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, greedy_rates, width, 
                   label='Greedy Agent', color=COLORS['greedy'], 
                   alpha=0.9, edgecolor='white', linewidth=1)
    bars2 = ax.bar(x_pos + width/2, mcts_rates, width, 
                   label='MCTS Agent', color=COLORS['mcts'], 
                   alpha=0.9, edgecolor='white', linewidth=1)
    
    # Customize chart
    ax.set_xlabel('Conditions', fontweight='bold')
    ax.set_ylabel('Judge Accuracy after Debate (%)', fontweight='bold')
    ax.set_title('Agent Type Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions, rotation=0, ha='center', fontsize=11)
    
    # Set y-axis limits for better zoom
    ax.set_ylim(y_min, y_max)
    
    # Enhanced grid with fixed scale (steps of 5)
    y_min_rounded = int(y_min // 5) * 5
    y_max_rounded = int((y_max + 4) // 5) * 5
    y_ticks = np.arange(y_min_rounded, y_max_rounded + 1, 5)
    ax.set_yticks(y_ticks)
    ax.set_ylim(y_min_rounded, y_max_rounded)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Enhanced legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Add value labels
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}%', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/graph3_agent_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def print_data_summary(debates_df, evaluations_df):
    """Print a summary of available data for analysis."""
    print("=== DATA SUMMARY ===")
    print(f"Total debate records: {len(debates_df)}")
    print(f"Total evaluation records: {len(evaluations_df)}")
    
    print("\n--- Debate Data Breakdown ---")
    print("Agent types:", debates_df['agent_type'].value_counts().to_dict())
    print("Pixel counts:", debates_df['pixels'].value_counts().to_dict())
    print("Precommit:", debates_df['precommit'].value_counts().to_dict())
    print("Starter:", debates_df['started'].value_counts().to_dict())
    
    print("\n--- Evaluation Data Breakdown ---")
    print("Pixel counts:", evaluations_df['pixels'].value_counts().to_dict())
    print("Judge baseline data available:", not evaluations_df.empty)
    print("Evaluations data:")
    print(evaluations_df[['pixels', 'accuracy']].to_string())
    
    print("\n--- Key Metrics ---")
    for k in [4, 6]:
        print(f"\nk={k} pixels:")
        
        # Judge baseline
        judge_data = evaluations_df[evaluations_df['pixels'] == k]
        if not judge_data.empty:
            judge_baseline = judge_data['accuracy'].mean()
            print(f"  Judge baseline (random): {judge_baseline:.3f} ({len(judge_data)} records)")
        else:
            print(f"  Judge baseline (random): NO DATA")
        
        # Greedy with precommit
        greedy_data = debates_df[
            (debates_df['pixels'] == k) & 
            (debates_df['agent_type'] == 'greedy') & 
            (debates_df['precommit'] == True)
        ]
        if not greedy_data.empty:
            greedy_precommit = greedy_data['accuracy'].mean()
            print(f"  Greedy with precommit: {greedy_precommit:.3f} ({len(greedy_data)} records)")
        else:
            print(f"  Greedy with precommit: NO DATA")
        
        # MCTS with precommit
        mcts_data = debates_df[
            (debates_df['pixels'] == k) & 
            (debates_df['agent_type'] == 'mcts') & 
            (debates_df['precommit'] == True)
        ]
        if not mcts_data.empty:
            mcts_precommit = mcts_data['accuracy'].mean()
            print(f"  MCTS with precommit: {mcts_precommit:.3f} ({len(mcts_data)} records)")
        else:
            print(f"  MCTS with precommit: NO DATA")

def main():
    parser = argparse.ArgumentParser(description='Analyze debate results and generate graphs')
    parser.add_argument('--output-dir', default='outputs/figures', help='Output directory for graphs')
    parser.add_argument('--summary-only', action='store_true', help='Only print data summary')
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    debates_df, evaluations_df, judges_df = load_data()
    
    # Print summary
    print_data_summary(debates_df, evaluations_df)
    
    if args.summary_only:
        return
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    print(f"\nGenerating graphs in {args.output_dir}/...")
    
    # Generate graphs
    try:
        create_graph1_judge_vs_debate(debates_df, evaluations_df, args.output_dir)
        print("✓ Graph 1 (Judge vs Debate) generated")
    except Exception as e:
        print(f"✗ Error generating Graph 1: {e}")
    
    try:
        create_graph2_precommit_impact(debates_df, args.output_dir)
        print("✓ Graph 2 (Precommit Impact) generated")
    except Exception as e:
        print(f"✗ Error generating Graph 2: {e}")
    
    try:
        create_graph3_agent_comparison(debates_df, args.output_dir)
        print("✓ Graph 3 (Agent Comparison) generated")
    except Exception as e:
        print(f"✗ Error generating Graph 3: {e}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
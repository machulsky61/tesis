"""
Centralized path configuration for the debate_mnist system.
This module defines all file and directory paths used throughout the system.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
CONFIGS_DIR = BASE_DIR / "configs"
DATA_DIR = BASE_DIR / "data"

# Output subdirectories
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"
DEBATES_VIZ_DIR = VISUALIZATIONS_DIR / "debates"
FIGURES_DIR = VISUALIZATIONS_DIR / "figures"

# CSV files
DEBATES_CSV = OUTPUTS_DIR / "debates.csv"
DEBATES_ASIMETRICOS_CSV = OUTPUTS_DIR / "debates_asimetricos.csv"
EVALUATIONS_CSV = OUTPUTS_DIR / "evaluations.csv"
JUDGES_CSV = OUTPUTS_DIR / "judges.csv"

# Ensure all directories exist
def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        MODELS_DIR,
        OUTPUTS_DIR,
        CONFIGS_DIR,
        VISUALIZATIONS_DIR,
        DEBATES_VIZ_DIR,
        FIGURES_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_model_path(judge_name):
    """Get the path for a judge model file."""
    return MODELS_DIR / f"{judge_name}.pth"

def get_debate_folder(debate_id, note=""):
    """Get the path for a debate visualization folder."""
    folder_name = f"debate_{debate_id}"
    if note:
        folder_name += f"_{note}"
    return DEBATES_VIZ_DIR / folder_name

def get_config_path(config_name):
    """Get the path for a configuration file."""
    if not config_name.endswith('.json'):
        config_name += '.json'
    return CONFIGS_DIR / config_name

def get_figure_path(figure_name):
    """Get the path for analysis figure files."""
    return FIGURES_DIR / figure_name

# Initialize directories when module is imported
ensure_directories()
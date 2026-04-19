"""
Global configuration for null model analysis pipeline.

Settings for:
- File paths (data, intermediate files, outputs)
- Analysis parameters (binning, sampling, metrics)
- Reproducibility (random seeds)
"""

from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
INTERMEDIATE_DIR = PROJECT_ROOT / 'data' / 'intermediate'  # Pre-computed graphs, binning models, etc.
OUTPUT_DIR = PROJECT_ROOT / 'outputs'

# Common intermediate files used between modules
DISTANCE_GRAPH_PATH = INTERMEDIATE_DIR / 'distance_graph.pt'
BINNING_MODEL_PATH = INTERMEDIATE_DIR / 'binning_model.pkl'

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

# Binning model
N_BINS = 20

# Null model sampling
N_NULLS = 100

# Reproducibility
RANDOM_SEED = 42

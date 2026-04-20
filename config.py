"""
Global path configuration for the jhapl-GNN repository.

Paths are absolute, resolved from this file's location, so callers don't
need to care about the current working directory.

    RAW_DATA_DIR     -> raw .pbz2 graph exports (neuron morphologies)
    INTERMEDIATE_DIR -> shared preprocessed artifacts (.pt, .pkl)
    OUTPUT_DIR       -> results root; each subpackage writes to OUTPUT_DIR / <name>
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

RAW_DATA_DIR     = REPO_ROOT.parent.parent / "demo_graph_exports"
INTERMEDIATE_DIR = REPO_ROOT / "intermediate_outputs"
OUTPUT_DIR       = REPO_ROOT / "outputs"


def output_dir(subpackage: str) -> Path:
    """Return the output subdirectory for a subpackage, creating it if needed."""
    path = OUTPUT_DIR / subpackage
    path.mkdir(parents=True, exist_ok=True)
    return path

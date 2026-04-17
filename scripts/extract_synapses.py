"""
Wrapper script to run jhapl_gnn.extract_synapses module from scripts/

For import and programmatic usage, use: from jhapl_gnn.extract_synapses import extract_synapses

Usage:
    python scripts/extract_synapses.py ../data/raw/graph_exports
    
Or use the module directly:
    python -m jhapl_gnn.extract_synapses ../data/raw/graph_exports \
        --distance-graph ../data/processed/distance_graph.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jhapl_gnn.extract_synapses import main

if __name__ == '__main__':
    main()

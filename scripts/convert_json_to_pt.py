"""
Convert existing JSON data files to PyTorch .pt format.

This utility helps migrate from JSON to PyTorch format for performance
and compatibility with ML pipelines.

Usage:
    # Convert both synapses and positions
    python3 convert_json_to_pt.py
    
    # Convert specific files
    python3 convert_json_to_pt.py --synapses data/processed/synapses.json --positions data/processed/positions.json
    
    # Convert with custom output paths
    python3 convert_json_to_pt.py --synapses data/processed/synapses.json --output-synapses data/processed/custom_synapses.pt
"""

import json
import torch
import argparse
import numpy as np
from pathlib import Path

# Default paths
DEFAULT_SYNAPSES_JSON = Path(__file__).parent.parent / 'data' / 'processed' / 'synapses.json'
DEFAULT_POSITIONS_JSON = Path(__file__).parent.parent / 'data' / 'processed' / 'positions.json'


def convert_synapses_json_to_pt(input_file, output_file=None):
    """Convert synapses JSON to PyTorch .pt format."""
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"✗ Input file not found: {input_path}")
        return False
    
    if output_file is None:
        output_file = input_path.parent / (input_path.stem + '.pt')
    
    output_path = Path(output_file)
    
    print(f"\nConverting synapses...")
    print(f"  Input:  {input_path.absolute()}")
    print(f"  Output: {output_path.absolute()}")
    
    try:
        # Load JSON
        with open(input_path, 'r') as f:
            synapses = json.load(f)
        
        print(f"  ✓ Loaded {len(synapses)} synapses from JSON")
        
        # Create PyTorch format
        synapses_data = {
            'synapses': synapses,
        }
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(synapses_data, output_path)
        
        print(f"  ✓ Saved to PyTorch format")
        print(f"    File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def convert_positions_json_to_pt(input_file, output_file=None):
    """Convert positions JSON to PyTorch .pt format."""
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"✗ Input file not found: {input_path}")
        return False
    
    if output_file is None:
        output_file = input_path.parent / (input_path.stem + '.pt')
    
    output_path = Path(output_file)
    
    print(f"\nConverting positions...")
    print(f"  Input:  {input_path.absolute()}")
    print(f"  Output: {output_path.absolute()}")
    
    try:
        # Load JSON
        with open(input_path, 'r') as f:
            positions_json = json.load(f)
        
        print(f"  ✓ Loaded positions for {len(positions_json)} neurons from JSON")
        
        # Convert to tensor format
        node_ids = sorted(list(positions_json.keys()))
        positions_array = np.array([positions_json[nid] for nid in node_ids])
        positions_tensor = torch.tensor(positions_array, dtype=torch.float)
        
        # Create PyTorch format
        positions_data = {
            'positions': positions_tensor,
            'node_ids': node_ids,
        }
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(positions_data, output_path)
        
        print(f"  ✓ Saved to PyTorch format")
        print(f"    Shape: {positions_tensor.shape}")
        print(f"    File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Convert JSON graph data to PyTorch .pt format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Convert both (default locations):
    python3 convert_json_to_pt.py
  
  Convert specific files:
    python3 convert_json_to_pt.py --synapses data/my_synapses.json --positions data/my_positions.json
  
  Convert with custom outputs:
    python3 convert_json_to_pt.py --synapses data/synapses.json --output-synapses data/synapses_new.pt
        """
    )
    
    parser.add_argument(
        '--synapses',
        type=str,
        default=None,
        help='Input synapses JSON file (default: data/processed/synapses.json)'
    )
    
    parser.add_argument(
        '--positions',
        type=str,
        default=None,
        help='Input positions JSON file (default: data/processed/positions.json)'
    )
    
    parser.add_argument(
        '--output-synapses',
        type=str,
        default=None,
        help='Output synapses .pt file (default: same as input with .pt extension)'
    )
    
    parser.add_argument(
        '--output-positions',
        type=str,
        default=None,
        help='Output positions .pt file (default: same as input with .pt extension)'
    )
    
    args = parser.parse_args()
    
    # Determine input files
    synapses_input = Path(args.synapses) if args.synapses else DEFAULT_SYNAPSES_JSON
    positions_input = Path(args.positions) if args.positions else DEFAULT_POSITIONS_JSON
    
    print("=" * 80)
    print("JSON to PyTorch Conversion Utility")
    print("=" * 80)
    
    success = True
    
    # Convert synapses
    if synapses_input.exists():
        success &= convert_synapses_json_to_pt(synapses_input, args.output_synapses)
    else:
        print(f"\n⊘ Synapses file not found (skipping): {synapses_input}")
    
    # Convert positions
    if positions_input.exists():
        success &= convert_positions_json_to_pt(positions_input, args.output_positions)
    else:
        print(f"\n⊘ Positions file not found (skipping): {positions_input}")
    
    print("\n" + "=" * 80)
    if success:
        print("✓ Conversion complete!")
    else:
        print("✗ Some conversions failed")
    print("=" * 80)


if __name__ == '__main__':
    main()

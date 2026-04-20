import time
import argparse
import os

from config import INTERMEDIATE_DIR, OUTPUT_DIR, RAW_DATA_DIR
from ADP.helper_functions import (
    generate_skeleton_data,
    build_kd_trees,
    calc_adp,
    build_global_kd_trees,
    convert_adp,
    build_graph)


def main():
    parser = argparse.ArgumentParser(
        description="ADP Pipeline - Command Line Driven"
    )

    parser.add_argument(
        "--raw_data_path",
        type=str,
        default=str(RAW_DATA_DIR),
        help=f"Path to neuron graph exports directory (default: {RAW_DATA_DIR})"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=str(OUTPUT_DIR / "ADP"),
        help=f"Path to output / checkpoint directory (default: {OUTPUT_DIR / 'ADP'})"
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=5000,
        help="Radius in nm for global KD-tree + ADP calculation (default: 5000)"
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="Minimum number of ADP co-travel distance(microns) for edge inclusion in ADP graph (default: 0)"
    )


    args = parser.parse_args()

    rel_neuron_graph_path = args.raw_data_path
    rel_data_checkpoint_path = args.output_path
    radius = args.radius
    threshold = args.threshold

    os.makedirs(rel_data_checkpoint_path, exist_ok=True)


    # Skeletonization
    time_start = time.time()
    generate_skeleton_data(rel_neuron_graph_path, rel_data_checkpoint_path)
    print(f"Skeletonization generation completed in {time.time() - time_start:.2f} seconds")


    # KD-tree construction
    time_start = time.time()
    build_kd_trees(rel_data_checkpoint_path)
    print(f"KD-tree generation completed in {time.time() - time_start:.2f} seconds")


    # Global KD-tree construction
    time_start = time.time()
    build_global_kd_trees(rel_data_checkpoint_path, radius)
    print(f"Global KD-tree generation completed in {time.time() - time_start:.2f} seconds")


    # ADP calculation
    time_start = time.time()
    calc_adp(rel_data_checkpoint_path, radius)
    print(f"ADP generation completed in {time.time() - time_start:.2f} seconds")


    # ADP neuron id conversion
    time_start = time.time()
    convert_adp(rel_data_checkpoint_path)
    print(f"ADP id conversion completed in {time.time() - time_start:.2f} seconds")

    # Graph Generation — final ADP graph lives in the shared intermediate dir.
    time_start = time.time()
    build_graph(rel_data_checkpoint_path, threshold, output_dir=str(INTERMEDIATE_DIR))
    print(f"Graph generation completed in {time.time() - time_start:.2f} seconds")

    # Clean up the pipeline's own intermediates (leave unrelated files alone)
    intermediates = [
        "skeletonization_data_simple.pkl",
        "neuron_ids.pkl",
        "kd_tree_data_simple.pkl",
        "dendrite_kd_tree.pkl",
        "dendrite_owner_dict.pkl",
        "axon_kd_tree.pkl",
        "axon_owner_dict.pkl",
        "neuron_to_idx.pkl",
        "idx_to_neuron.pkl",
        "adp_dict.pkl",
        "adp_data.pkl",
    ]
    for filename in intermediates:
        file_path = os.path.join(rel_data_checkpoint_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted {file_path}")


if __name__ == "__main__":
    main()
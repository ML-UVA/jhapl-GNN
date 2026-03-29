import time
import argparse

from skeletonization_gen_simple import generate_skeleton_data
from kd_tree_construction_simple import build_kd_trees
from adp_calc_simple import calc_adp
from build_global_kd_tree_simple import build_global_kd_trees
from convert_adp_neuron_ids import convert_adp


def main():
    parser = argparse.ArgumentParser(
        description="ADP Pipeline - Command Line Driven"
    )

    parser.add_argument(
        "--graph_path",
        type=str,
        required=True,
        help="Relative path to neuron graph exports directory"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Relative path to data checkpoint directory"
    )

    parser.add_argument(
        "--radius",
        type=float,
        required=True,
        help="Radius in nm for global KD-tree + ADP calculation"
    )

    args = parser.parse_args()

    rel_neuron_graph_path = args.graph_path
    rel_data_checkpoint_path = args.data_path
    radius = args.radius


    # Skeletonization
    time_start = time.time()
    generate_skeleton_data(rel_neuron_graph_path)
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


if __name__ == "__main__":
    main()
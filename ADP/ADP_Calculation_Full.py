import time
import argparse
import os

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
        "--graph_path",
        type=str,
        required=True,
        help="Path to neuron graph exports directory (relative to current working directory)"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to data checkpoint directory (relative to current working directory)"
    )

    parser.add_argument(
        "--radius",
        type=float,
        required=True,
        help="Radius in nm for global KD-tree + ADP calculation"
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="Minimum number of ADP co-travel distance(microns) for edge inclusion in ADP graph (default: 0)"
    )


    args = parser.parse_args()

    rel_neuron_graph_path = args.graph_path
    rel_data_checkpoint_path = args.data_path
    radius = args.radius
    threshold = args.threshold



    # Skeletonization
    time_start = time.time()
    generate_skeleton_data(rel_neuron_graph_path,rel_data_checkpoint_path)
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

    # Graph Generation
    time_start = time.time()
    build_graph(rel_data_checkpoint_path, threshold)
    print(f"Graph generation completed in {time.time() - time_start:.2f} seconds")

    # DELETE everything except .pt files
    for filename in os.listdir(rel_data_checkpoint_path):
        file_path = os.path.join(rel_data_checkpoint_path, filename)

        if os.path.isfile(file_path) and not filename.endswith(".pt"):
            os.remove(file_path)


if __name__ == "__main__":
    main()
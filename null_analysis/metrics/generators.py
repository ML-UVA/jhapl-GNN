"""
Null model metric generation pipeline.

Runs arbitrary metric functions on graphs generated from null model
generators and compares their distributions to ground truth values.
"""

import warnings
import networkx as nx
import pandas as pd
import numpy as np

# Suppress pandas FutureWarnings about concat behavior
warnings.filterwarnings('ignore', category=FutureWarning)


def run_null_models(null_generators: list, metrics: list, GT: nx.Graph, N=50):
    """
    Generate null model graphs and compute metrics for each.

    Creates N instances of each null model generator applied to ground truth GT,
    then evaluates all provided metric functions on each generated graph.

    Parameters
    ----------
    null_generators : list of callable
        Functions taking GT as input and returning a null model graph.

    metrics : list of callable
        Functions taking a graph as input and returning a single numeric value.

    GT : networkx.Graph or networkx.DiGraph
        Ground truth graph to base null models on.

    N : int, optional
        Number of null model instances per generator. Default: 50.

    Returns
    -------
    dict
        Keys: null generator function names.
        Values: list of lists, each sublist has metric values for one instance.
    """
    results = {
        null_gen.__name__: [] for null_gen in null_generators
    }

    for _ in range(N):
        for null_gen in null_generators:
            G = null_gen(GT)
            metric_values = []
            for metric in metrics:
                # Handle triangles metric on directed graphs
                if metric.__name__ == 'triangles' and G.is_directed():
                    metric_values.append(metric(G.to_undirected()))
                else:
                    metric_values.append(metric(G))
            results[null_gen.__name__].append(metric_values)
    return results


def summarize_results(GT: nx.Graph, results: dict, metrics: list):
    """
    Summarize null model results with ground truth comparison.

    Computes mean and standard deviation of each metric across null model
    samples and compares to ground truth values.

    Parameters
    ----------
    GT : networkx.Graph or networkx.DiGraph
        Ground truth graph.

    results : dict
        Output from run_null_models(). Maps generator names to metric lists.

    metrics : list of callable
        Metric functions used (for naming columns and computing GT values).

    Returns
    -------
    pd.DataFrame
        Rows: ground truth plus one row per null model generator.
        Columns: 'model' name plus for each metric:
        'mean {metric_name}' and 'stdev {metric_name}'.
    """
    true_res = pd.DataFrame({
        "model": ["ground truth"]
    })
    for metric in metrics:
        # Handle triangles metric on directed graphs
        if metric.__name__ == 'triangles' and GT.is_directed():
            true_res[f"mean {metric.__name__}"] = [metric(GT.to_undirected())]
        else:
            true_res[f"mean {metric.__name__}"] = [metric(GT)]
        true_res[f"stdev {metric.__name__}"] = [None]

    summary = pd.DataFrame({
        "model": list(results.keys())
    })
    for i, metric in enumerate(metrics):
        summary[f"mean {metric.__name__}"] = [
            np.mean([x[i] for x in results[m]]) for m in results
        ]
        summary[f"stdev {metric.__name__}"] = [
            np.std([x[i] for x in results[m]]) for m in results
        ]
    summary = pd.concat([true_res, summary], ignore_index=True)
    return summary
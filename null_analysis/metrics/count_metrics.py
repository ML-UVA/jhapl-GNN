"""
Triadic motif counting and analysis utilities.

This module provides functions to compute triadic census (3-node motif counts)
for empirical networks, compare against null models, and visualize results.
"""

from networkx.algorithms.triads import triadic_census
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def count_tri(G):
    """
    Compute triadic census for a given directed graph.

    Enumerates all 16 possible 3-node directed subgraph types and counts
    their occurrences in the input graph using NetworkX's triadic_census.

    Parameters
    ----------
    G : networkx.DiGraph
        Input directed graph.

    Returns
    -------
    pd.DataFrame
        Motif names indexed as rows, 'observed' counts in a single column.
    """
    obs = triadic_census(G)
    obs_df = (
        pd.DataFrame(obs.items(), columns=["motif", "observed"])
        .set_index("motif")
    )
    return obs_df


def motif_distribution(G, null_fn, n):
    """
    Generate distribution of motif counts from null model samples.

    Applies a null model generator function n times to the input graph,
    computing triadic census for each null network. Useful for assessing
    significance of observed motif counts.

    Parameters
    ----------
    G : networkx.DiGraph
        Input graph to generate null models from.

    null_fn : callable
        Function that takes G and returns a null model graph.

    n : int
        Number of null model samples to generate.

    Returns
    -------
    pd.DataFrame
        Motif counts across all samples (n rows × 16 motif columns).
    """
    records = []
    for _ in range(n):
        G_null = null_fn(G)
        records.append(triadic_census(G_null))
    return pd.DataFrame(records)


def generate_motif_df(GT, null_functions, n=5):
    """
    Compare observed motif counts to multiple null models with z-scores.

    For each null model, computes mean and standard deviation of motif
    distributions, then calculates z-scores to assess which motifs are
    significantly enriched or depleted relative to chance.

    Parameters
    ----------
    GT : networkx.DiGraph
        Ground truth (empirical) directed graph.

    null_functions : list of callable
        List of null model generator functions, each taking GT as input.

    n : int, optional
        Number of null model samples per generator. Default: 5.

    Returns
    -------
    pd.DataFrame
        Index: motif names.
        Columns: 'observed' plus for each null model:
        '{name}_mean', '{name}_std', '{name}_z' (z-score).
    """
    obs_df = count_tri(GT)
    summary = obs_df.copy()

    nulls = {
        null_f.__name__: motif_distribution(GT, null_f, n) for null_f in null_functions
    }

    for name, df_null in nulls.items():
        summary[f"{name}_mean"] = df_null.mean()
        summary[f"{name}_std"]  = df_null.std()
        summary[f"{name}_z"]    = (
            summary["observed"] - summary[f"{name}_mean"]
        ) / summary[f"{name}_std"]
    return summary


def plot_summary(summary, nulls):
    """
    Visualize observed vs null model motif counts in a bar chart.

    Plots observed motif counts alongside mean counts from multiple null
    models, with log scale on y-axis for visibility across orders of magnitude.

    Parameters
    ----------
    summary : pd.DataFrame
        Output from generate_motif_df() with 'observed' and null model columns.

    nulls : list of callable
        Null model generator functions (used only for labels via __name__).

    Returns
    -------
    None
        Displays matplotlib figure.
    """
    motifs = summary.index
    x = np.arange(len(motifs))
    width = 0.25

    plt.figure(figsize=(14, 5))

    plt.bar(x - width, summary["observed"], width, label="Observed")

    i = 0
    for null_f in nulls:
        plt.bar(
            x + i * width,
            summary[f"{null_f.__name__}_mean"],
            width,
            label=null_f.__name__
        )
        i += 1

    plt.xticks(x, motifs, rotation=45)
    plt.ylabel("Motif count (log scale)")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()
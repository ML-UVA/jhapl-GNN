import numpy as np
import pandas as pd


class BinModel:
    """
    Stores bin structure and empirical edge probabilities.

    Attributes
    ----------
    bin_edges : np.ndarray
        Edges of ADP bins.
    bin_probs : np.ndarray
        Empirical P(edge | ADP in bin).
    bin_centers : np.ndarray
        Mean ADP value in each bin.
    """

    def __init__(self, bin_edges, bin_probs, bin_centers):
        self.bin_edges = bin_edges
        self.bin_probs = bin_probs
        self.bin_centers = bin_centers

    def lookup_prob(self, adp):
        """
        Return P(edge | ADP) using bin lookup.
        """
        k = np.digitize(adp, self.bin_edges) - 1
        k = np.clip(k, 0, len(self.bin_probs) - 1)
        return self.bin_probs[k]


def compute_bins(adp_values, edge_indicator, n_bins=20, method="quantile"):
    """
    Compute empirical P(edge | ADP) using binning.

    Parameters
    ----------
    adp_values : array-like
        ADP values for all measurable node pairs.

    edge_indicator : array-like
        1 if edge exists, 0 otherwise.

    n_bins : int
        Number of bins.

    method : str
        'quantile' or 'uniform'

    Returns
    -------
    BinModel
    """

    adp_values = np.asarray(adp_values)
    edge_indicator = np.asarray(edge_indicator)

    df = pd.DataFrame({
        "adp": adp_values,
        "edge": edge_indicator
    })

    if method == "quantile":
        df["bin"] = pd.qcut(df["adp"], q=n_bins, duplicates="drop")
    elif method == "uniform":
        df["bin"] = pd.cut(df["adp"], bins=n_bins)
    else:
        raise ValueError("method must be 'quantile' or 'uniform'")

    grouped = df.groupby("bin")

    bin_probs = grouped["edge"].mean().values
    bin_centers = grouped["adp"].mean().values

    # extract edges from pandas Interval objects
    bin_edges = np.array([interval.left for interval in grouped.groups.keys()])
    last_edge = list(grouped.groups.keys())[-1].right
    bin_edges = np.append(bin_edges, last_edge)

    return BinModel(bin_edges, bin_probs, bin_centers)


def assign_bins(adp_values, bin_model):
    """
    Assign each ADP value to a bin index.

    Returns
    -------
    np.ndarray
        Bin indices.
    """
    bins = np.digitize(adp_values, bin_model.bin_edges) - 1
    bins = np.clip(bins, 0, len(bin_model.bin_probs) - 1)
    return bins
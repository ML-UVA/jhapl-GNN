import numpy as np
import pandas as pd


class BinModel:
    """
    Stores bin structure and empirical edge probabilities for a spatial feature.

    Attributes
    ----------
    bin_edges : np.ndarray
        Edges of feature bins.
    bin_probs : np.ndarray
        Empirical P(edge | feature in bin).
    bin_centers : np.ndarray
        Mean feature value in each bin.
    """

    def __init__(self, bin_edges, bin_probs, bin_centers):
        self.bin_edges = bin_edges
        self.bin_probs = bin_probs
        self.bin_centers = bin_centers

    def lookup_prob(self, feature):
        """
        Return P(edge | feature) using bin lookup.

        Parameters
        ----------
        feature : float or array-like
            Feature value(s) to look up.

        Returns
        -------
        float or np.ndarray
            Empirical edge probability for the given feature value(s).
        """
        k = np.digitize(feature, self.bin_edges) - 1
        k = np.clip(k, 0, len(self.bin_probs) - 1)
        return self.bin_probs[k]


def compute_bins(feature_values, edge_indicator, n_bins=20, method="quantile"):
    """
    Compute empirical P(edge | feature) using binning.

    For each bin of a continuous spatial feature, estimates the probability
    that a node pair is connected based on observed data.

    Parameters
    ----------
    feature_values : array-like of shape (n_pairs,)
        Feature values (e.g., distance, ADP) for all measurable node pairs.

    edge_indicator : array-like of shape (n_pairs,)
        1 if edge exists, 0 otherwise.

    n_bins : int, optional
        Number of bins. Default: 20.

    method : str, optional
        Binning method: 'quantile' or 'uniform'. Default: 'quantile'.

    Returns
    -------
    BinModel
        Object storing bin edges, bin-wise edge probabilities, and bin centers.
    """

    feature_values = np.asarray(feature_values)
    edge_indicator = np.asarray(edge_indicator)

    df = pd.DataFrame({
        "feature": feature_values,
        "edge": edge_indicator
    })

    if method == "quantile":
        df["bin"] = pd.qcut(df["feature"], q=n_bins, duplicates="drop")
    elif method == "uniform":
        df["bin"] = pd.cut(df["feature"], bins=n_bins)
    else:
        raise ValueError("method must be 'quantile' or 'uniform'")

    grouped = df.groupby("bin")

    bin_probs = grouped["edge"].mean().values
    bin_centers = grouped["feature"].mean().values

    # extract edges from pandas Interval objects
    bin_edges = np.array([interval.left for interval in grouped.groups.keys()])
    last_edge = list(grouped.groups.keys())[-1].right
    bin_edges = np.append(bin_edges, last_edge)

    return BinModel(bin_edges, bin_probs, bin_centers)


def assign_bins(feature_values, bin_model):
    """
    Assign each feature value to a bin index.

    Parameters
    ----------
    feature_values : array-like
        Feature values to assign to bins.

    bin_model : BinModel
        Binning model from compute_bins().

    Returns
    -------
    np.ndarray
        Bin indices for each feature value.
    """
    bins = np.digitize(feature_values, bin_model.bin_edges) - 1
    bins = np.clip(bins, 0, len(bin_model.bin_probs) - 1)
    return bins
from networkx.algorithms.triads import triadic_census
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
def count_tri(G):
    obs = triadic_census(G)
    obs_df = (
        pd.DataFrame(obs.items(), columns=["motif", "observed"])
        .set_index("motif")
    )
    return obs_df

def motif_distribution(G, null_fn, n):
    records = []
    for _ in range(n):
        G_null = null_fn(G)
        records.append(triadic_census(G_null))
    return pd.DataFrame(records)

def generate_motif_df(GT, null_functions, n=5):
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
    motifs = summary.index
    x = np.arange(len(motifs))
    width = 0.25

    plt.figure(figsize=(14,5))

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
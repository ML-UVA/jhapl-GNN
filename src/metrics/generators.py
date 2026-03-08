import networkx as nx
import pandas as pd
import numpy as np
"""
Pipeline for running an arbitrary amount of metric functions on null models against a ground truth.
"""

def run_null_models(null_generators: list, metrics: list, GT: nx.Graph, N=50):
    results = {
        null_gen.__name__: [] for null_gen in null_generators
    }

    for _ in range(N):
        for null_gen in null_generators:
            G = null_gen(GT)
            results[null_gen.__name__].append([
                metric(G) for metric in metrics
            ])
    return results

def summarize_results(GT: nx.Graph, results: list[list[float]], metrics: list):
    true_res = pd.DataFrame({
        "model": "ground truth"
    })
    for metric in metrics:
        true_res[f"mean {metric.__name__}"] = metric(GT)
        true_res[f"stdev {metric.__name__}"] = None
    summary = pd.DataFrame({
    "model": list(results.keys())
    })
    for i, metric in enumerate(metrics):
        summary[f"mean {metric.__name__}"] = [np.mean([x[i*2] for x in results[m]]) for m in results]
        summary[f"stdev {metric.__name__}"] = [np.std([x[i*2+1] for x in results[m]]) for m in results]
    summary = pd.concat([true_res, summary], ignore_index=True)
    return summary
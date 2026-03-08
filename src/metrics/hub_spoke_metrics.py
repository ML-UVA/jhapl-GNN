import numpy as np
import networkx as nx

def gini(G):
    x = np.array([d for _, d in G.degree()])
    x = np.sort(np.array(x))
    n = len(x)
    return (2*np.sum((np.arange(n)+1)*x) / (n*np.sum(x))) - (n+1)/n

def coef_variation(G):
    x = np.array([d for _, d in G.degree()])
    return x.std() / x.mean()

def mean_deg(G):
    x = np.array([d for _, d in G.degree()])
    return np.mean(x)

def max_deg(G):
    x = np.array([d for _, d in G.degree()])
    return np.max(x)

def deg_assortativity(G):
    return nx.degree_assortativity_coefficient(G)
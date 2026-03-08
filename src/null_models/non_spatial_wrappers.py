from non_spatial_null_models import *
import networkx as nx
from src.config import RANDOM_SEED

def ER(GT: nx.Graph):
    n = GT.number_of_nodes()
    m = GT.number_of_edges()
    p = (2*m) / (n*(n-1))
    return erdos_renyi_directed(n, p, RANDOM_SEED)

def configuration(GT: nx.Graph):
    degree_seq = [d for _, d in GT.degree()]
    return configuration_model_directed(degree_seq, degree_seq, RANDOM_SEED)

def BA(GT: nx.Graph):
    n = GT.number_of_nodes()
    m = GT.number_of_edges()
    return barabasi_albert_directed(n, m, RANDOM_SEED)

def smallworld(GT: nx.Graph):
    n = GT.number_of_nodes()
    m = GT.number_of_edges()
    p = (2*m) / (n*(n-1))
    k = len(GT.degree())
    return watts_strogatz_directed(n, k, p, RANDOM_SEED)
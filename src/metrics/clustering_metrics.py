import networkx as nx


def clustering(G):
    return nx.average_clustering(G)

def transitivity(G):
    return nx.transitivity(G)

def triangles(G):
    ttt = nx.triangles(G)
    s = 0
    for n in ttt:
        s += ttt[n]
    return s/len(ttt)

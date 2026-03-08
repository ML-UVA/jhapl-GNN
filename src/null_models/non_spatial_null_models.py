import random

def erdos_renyi_directed(n, p, seed=None, self_loops=False):
    if seed is not None:
        random.seed(seed)

    adj = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(n):
            if not self_loops and i == j:
                continue
            if random.random() < p:
                adj[i].append(j)

    return adj

import random

def configuration_model_directed(in_degrees, out_degrees, seed=None):
    if seed is not None:
        random.seed(seed)

    if sum(in_degrees) != sum(out_degrees):
        raise ValueError("In-degree sum must equal out-degree sum")

    n = len(in_degrees)
    in_stubs = []
    out_stubs = []

    for i in range(n):
        in_stubs.extend([i] * in_degrees[i])
        out_stubs.extend([i] * out_degrees[i])

    random.shuffle(in_stubs)
    random.shuffle(out_stubs)

    adj = {i: [] for i in range(n)}

    for u, v in zip(out_stubs, in_stubs):
        adj[u].append(v)

    return adj

import random

def barabasi_albert_directed(n, m, seed=None):
    if seed is not None:
        random.seed(seed)

    adj = {i: [] for i in range(m)}

    for i in range(m):
        for j in range(m):
            if i != j:
                adj[i].append(j)

    in_degree_list = []
    for i in range(m):
        in_degree_list.extend([i] * len(adj[i]))

    for new_node in range(m, n):
        adj[new_node] = []

        targets = set()
        while len(targets) < m:
            targets.add(random.choice(in_degree_list))

        for t in targets:
            adj[new_node].append(t)

        in_degree_list.extend(list(targets))
        in_degree_list.extend([new_node] * m)

    return adj

def watts_strogatz_directed(n, k, p, seed=None):
    if seed is not None:
        random.seed(seed)

    if k % 2 != 0:
        raise ValueError("k must be even")

    adj = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(1, k // 2 + 1):
            adj[i].append((i + j) % n)

    for i in range(n):
        for j in list(adj[i]):
            if random.random() < p:
                adj[i].remove(j)
                new = random.choice(
                    [x for x in range(n) if x != i and x not in adj[i]]
                )
                adj[i].append(new)

    return adj

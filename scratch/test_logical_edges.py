import networkx as nx
from core.data_loader import BASE_TOPOLOGY, TOPOLOGY_POS

def is_hidden(node):
    s = str(node)
    return s.islower() and s.isalpha()

def get_logical_edges():
    visible_nodes = [n for n in TOPOLOGY_POS.keys() if not is_hidden(n)]
    G = nx.DiGraph(BASE_TOPOLOGY)
    
    logical_edges = []
    for u in visible_nodes:
        # Find all v in visible_nodes reachable from u
        for v in visible_nodes:
            if u == v: continue
            try:
                # Find all simple paths? No, let's find the shortest path
                # and check if it only passes through hidden nodes (except u, v)
                path = nx.shortest_path(G, u, v)
                if all(is_hidden(node) for node in path[1:-1]):
                    logical_edges.append((u, v))
            except nx.NetworkXNoPath:
                continue
    return logical_edges

if __name__ == "__main__":
    le = get_logical_edges()
    print(f"Total logical edges: {len(le)}")
    for u, v in le[:10]:
        print(f"  {u} -> {v}")

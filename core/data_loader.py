import pandas as pd
import networkx as nx

def normalize_node(val):
    if val is None: return ""
    # Convert to string, strip spaces, but KEEP dots for Roman numerals
    return str(val).strip()

def load_network_data(df_req, df_cap):
    """
    Загружает данные из DataFrames и гарантирует связность графа.
    """
    requests = {}
    capacities = {}
    nodes_set = set()

    # 1. Parse Requests (Table 1.1)
    total_requested = 0
    for _, row in df_req.iterrows():
        src = normalize_node(row['Источник потока'])
        dst = normalize_node(row['Потребитель'])
        try:
            # Handle possible formatting in numeric strings
            val_str = str(row['Поток, кВт']).replace(',', '.').replace(' ', '')
            vol = float(val_str)
            if src and dst:
                requests[(src, dst)] = vol
                nodes_set.update([src, dst])
                total_requested += vol
        except (ValueError, TypeError): continue

    # 2. Parse Capacities (Table 1.2)
    for _, row in df_cap.iterrows():
        u = normalize_node(row['начало'])
        v = normalize_node(row['окончание'])
        try:
            val_str = str(row['Допустимая мощность']).replace(',', '.').replace(' ', '')
            cap = float(val_str)
            if u and v:
                capacities[(u, v)] = cap
                nodes_set.update([u, v])
        except (ValueError, TypeError): continue

    # 3. Guarantee Connectivity (Virtual Edge Rule)
    G = nx.DiGraph()
    G.add_nodes_from(nodes_set)
    for (u, v), cap in capacities.items():
        G.add_edge(u, v)

    virtual_cap = total_requested + 1000.0
    missing_paths = []

    for (src, dst) in requests.keys():
        if not nx.has_path(G, src, dst):
            # Если пути нет, добавляем прямое виртуальное ребро (ТЗ)
            capacities[(src, dst)] = virtual_cap
            G.add_edge(src, dst)
            missing_paths.append(f"{src}->{dst}")

    if missing_paths:
        print(f"Warning: {len(missing_paths)} paths not found in topology. Added virtual edges.")
        if len(missing_paths) > 0:
            print(f"Sample: {', '.join(missing_paths[:5])}...")

    nodes = list(nodes_set)
    destinations = list(set(dst for src, dst in requests.keys()))
    
    adj = {u: [] for u in nodes}
    for (u, v) in capacities.keys():
        adj[u].append(v)

    return nodes, destinations, adj, capacities, requests
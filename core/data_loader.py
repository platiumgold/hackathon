import pandas as pd
import networkx as nx

def normalize_node(val):
    if val is None: return ""
    # Сохраняем точки для римских цифр, убираем только лишние пробелы
    return str(val).strip()

def load_network_data(df_req, df_cap):
    """
    Загружает данные «как есть». Пути определяются только содержимым df_cap.
    """
    requests = {}
    capacities = {}
    nodes_set = set()

    # 1. Парсинг Заявок (Table 1.1)
    for _, row in df_req.iterrows():
        src = normalize_node(row['Источник потока'])
        dst = normalize_node(row['Потребитель'])
        try:
            val_str = str(row['Поток, кВт']).replace(',', '.').replace(' ', '')
            vol = float(val_str)
            if src and dst:
                requests[(src, dst)] = vol
                nodes_set.update([src, dst])
        except (ValueError, TypeError): continue

    # 2. Парсинг Ограничений (Table 1.2)
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

    nodes = list(nodes_set)
    destinations = list(set(dst for src, dst in requests.keys()))
    
    adj = {u: [] for u in nodes}
    for (u, v) in capacities.keys():
        if u in adj:
            adj[u].append(v)
        else:
            # Если узла из таблицы 1.2 нет в общем списке (например, он не в 1.1)
            adj[u] = [v]

    return nodes, destinations, adj, capacities, requests
import pandas as pd


def load_network_data(requests_df, capacities_df):
    """
    Преобразует сырые DataFrame из таблиц 1.1 и 1.2 во внутренние структуры графа.
    """
    requests = {}
    capacities = {}
    nodes_set = set()

    # Парсинг таблицы 1.1 (Заявки)
    for _, row in requests_df.iterrows():
        src = str(row['Источник потока']).strip()
        dst = str(row['Потребитель']).strip()
        # Игнорируем строки "Итого"
        if "Итого" in dst or "ВСЕГО" in src:
            continue
        try:
            vol = float(str(row['Поток, кВт']).replace(',', '').replace(' ', ''))
            requests[(src, dst)] = vol
            nodes_set.update([src, dst])
        except ValueError:
            continue  # Пропуск зашумленных строк

    # Парсинг таблицы 1.2 (Ограничения)
    for _, row in capacities_df.iterrows():
        u = str(row['начало']).strip()
        v = str(row['окончание']).strip()
        try:
            cap = float(str(row['Допустимая мощность']).replace(',', '').replace(' ', ''))
            capacities[(u, v)] = cap
            nodes_set.update([u, v])
        except ValueError:
            continue

    nodes = list(nodes_set)
    destinations = list(set(dst for src, dst in requests.keys()))

    adj_sets = {u: set() for u in nodes}
    for (u, v) in capacities.keys():
        adj_sets[u].add(v)

    adj = {u: list(adj_sets[u]) for u in adj_sets}

    return nodes, destinations, adj, capacities, requests
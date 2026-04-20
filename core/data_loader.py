import pandas as pd
import networkx as nx

# =================================================================
# КОНСТАНТНАЯ ТОПОЛОГИЯ (СХЕМА 1 - ALPHA)
# Этот каркас зафиксирован и не меняется. 
# На него накладываются ограничения из загружаемых файлов.
# =================================================================
#BASE_TOPOLOGY = [('A', 'XVIII.'), ('XVIII.', '6'), ('P', 'XI.'), ('XI.', '4'), ('F', 'II.'), ('II.', 'I.'), ('I.', '1'), ('B', 'X.'), ('X.', '10')]
BASE_TOPOLOGY = [
    ('A', 'XVIII.'), ('B', 'X.'), ('C', 'X.'), ('D', 'XV.'), ('E', 'XIV.'),
    ('F', 'II.'), ('G', 'II.'), ('H', 'III.'), ('I', 'V.'), ('J', 'VIII.'),
    ('K', 'XVI.'), ('L', 'XVI.'), ('M', 'X.'), ('N', 'VII.'), ('O', 'VII.'), ('P', 'XI.'),
    ('XVIII.', '6'), ('XVIII.', '7'), ('XVIII.', '5'), ('XVIII.', 'XI.'),
    ('XI.', '9'), ('XI.', '4'), ('XI.', '11'), ('XI.', 'X.'),
    ('X.', '10'), ('X.', '12'), ('X.', 'IX.'),
    ('IX.', '8'), ('IX.', 'XV.'),
    ('XV.', '29'), ('XV.', 'XIV.'),
    ('XIV.', '23'), ('XIV.', '22'), ('XIV.', '28'), ('XIV.', 'XII.'),
    ('XII.', '21'), ('XII.', '20'), ('XII.', '3'), ('XII.', '17'), ('XII.', '19'), ('XII.', 'VI.'), ('XII.', 'XIII.'),
    ('XIII.', '13'), ('XIII.', '18'), ('XIII.', '24'), ('XIII.', '14'), ('XIII.', '25'),
    ('VI.', '27'), ('VI.', '35'), ('VI.', '36'), ('VI.', '16'), ('VI.', 'VII.'), ('VI.', 'V.'),
    ('VII.', 'VI.'),
    ('V.', '32'), ('V.', 'VIII.'),
    ('VIII.', '37'), ('VIII.', 'XVII.'),
    ('XVII.', '33'), ('XVII.', '34'), ('XVII.', 'XVI.'),
    ('XVI.', '1'), ('XVI.', '2'), ('XVI.', '38'), ('XVI.', 'IV.'),
    ('IV.', 'III.'), ('III.', 'IV.'), ('III.', '30'), ('III.', '31'), ('III.', 'I.'),
    ('I.', 'II.'), ('II.', 'I.'), ('I.', 'L.'), ('L.', 'IX.')
]

def normalize_node(val):
    if val is None: return ""
    return str(val).strip()

def load_network_data(df_req, df_cap):
    """
    Загружает данные, накладывая ограничения на жестко заданный каркас сети.
    """
    requests = {}
    nodes_set = set()
    
    # Собираем все узлы из базы
    for u, v in BASE_TOPOLOGY:
        nodes_set.update([u, v])

    # 1. Парсинг Заявок (Table 1.1)
    total_requested = 0
    for _, row in df_req.iterrows():
        src = normalize_node(row['Источник потока'])
        dst = normalize_node(row['Потребитель'])
        try:
            val_str = str(row['Поток, кВт']).replace(',', '.').replace(' ', '')
            vol = float(val_str)
            if src and dst:
                requests[(src, dst)] = vol
                nodes_set.update([src, dst])
                total_requested += vol
        except (ValueError, TypeError): continue

    # 2. Формирование базовых мощностей (Бесконечность для каркаса)
    # Используем значение больше суммы всех заявок
    UNLIMITED_CAP = total_requested + 10000.0
    final_capacities = {}
    for u, v in BASE_TOPOLOGY:
        final_capacities[(u, v)] = UNLIMITED_CAP

    # 3. Наложение ограничений из файла (Table 1.2)
    for _, row in df_cap.iterrows():
        u = normalize_node(row['начало'])
        v = normalize_node(row['окончание'])
        try:
            val_str = str(row['Допустимая мощность']).replace(',', '.').replace(' ', '')
            cap = float(val_str)
            if u and v:
                # Если ребро есть в базе - обновляем (накладываем ограничение)
                # Если ребра нет в базе - добавляем как новое
                final_capacities[(u, v)] = cap
                nodes_set.update([u, v])
        except (ValueError, TypeError): continue

    nodes = list(nodes_set)
    destinations = list(set(dst for src, dst in requests.keys()))
    
    adj = {u: [] for u in nodes}
    for (u, v) in final_capacities.keys():
        if u in adj:
            adj[u].append(v)
        else:
            adj[u] = [v]

    return nodes, destinations, adj, final_capacities, requests
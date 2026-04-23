import pandas as pd
import networkx as nx

BASE_TOPOLOGY = [
    ('A', 'XVIII.'), ('B', 'X.'), ('C', 'X.'), ('D', 'XV.'), ('E', 'XIV.'),
    ('F', 'II.'), ('G', 'II.'), ('H', 'III.'), ('I', 'V.'), ('J', 'VIII.'),
    ('K', 'XVI.'),
    ('M', 'X.'), ('N', 'VII.'), ('O', 'VII.'), ('P', 'XI.'),

    # НИЖНЯЯ МАГИСТРАЛЬ
    ('XVIII.', 'g'), ('g', '5'), ('g', '6'), ('6', '7'),
    ('XVIII.', 'h'), ('h', '4'), ('h', 'i'), ('i', '9'), ('i', 'XI.'),
    ('XI.', 'j'), ('j', '11'), ('j', 'k'), ('k', '10'), ('k', 'X.'),
    ('X.', '12'), ('X.', 'IX.'),
    ('IX.', 'l'), ('l', '8'), ('l', 'XV.'),

    ('XIV.', '29'),
    ('XV.', 'XIV.'),

    # ЦЕПОЧКА XIV -> XII
    ('XIV.', 'm'), ('m', '23'),
    ('m', 'n'), ('n', '26'),
    ('n', 'o'), ('o', '22'),
    ('o', 'p'), ('p', '21'),
    ('p', 'q'), ('q', '28'),
    ('q', 'XII.'),

    ('XII.', '3'), ('XII.', 'XIII.'),

    ('XIII.', '17'),

    ('XIII.', '13'), ('XIII.', '18'), ('XIII.', '24'),
    ('13', '14'), ('13', '25'),

    # ЦЕПОЧКА XII -> VI
    ('XII.', 'r'), ('r', '20'),
    ('r', 's'), ('s', '19'),
    ('s', 'VI.'),

    # ВЕРХНЯЯ МАГИСТРАЛЬ
    ('VII.', 'VI.'),
    ('VI.', '27'), ('VI.', '15'),
    ('15', '16'),

    # Цепочка из VI для 35 и 36
    ('VI.', 't'), ('t', '35'),
    ('t', 'u'), ('u', '36'),

    ('V.', 'u'),

    ('V.', 'VIII.'), ('VIII.', '37'),

    # ИЗМЕНЕНО НАПРАВЛЕНИЕ: IX. -> I.
    ('II.', 'I.'), ('IX.', 'I.'),

    ('I.', 'a'), ('a', 'b'), ('b', 'III.'),
    ('a', '31'), ('b', '30'),

    ('III.', 'c'), ('c', 'd'), ('d', 'IV.'),
    ('c', '34'), ('d', '33'),

    ('XVI.', 'x'), ('L', 'x'),
    ('x', 'y'),
    ('y', '38'), ('y', '1'),
    ('1', '2'),

    ('XVII.', 'IV.'),
    ('IV.', 'XVI.'),

    ('V.', 'e'), ('e', '32'), ('e', 'XVII.')
]

# Координаты
TOPOLOGY_POS = {
    # Римские цифры
    'XIII.': (-11, 0), 'XII.': (-8.5, 0), 'XIV.': (-5, 0), 'XV.': (-2, 0),
    'IX.': (1.5, 0), 'X.': (4.5, 0), 'XI.': (9, 0), 'XVIII.': (13, 0),
    'VII.': (-14, 6), 'VI.': (-11.5, 6), 'V.': (-7, 6), 'VIII.': (-7, 7.5),
    'XVII.': (-3, 6), 'III.': (3, 6), 'IV.': (1, 7.5), 'XVI.': (1, 9.5),
    'I.': (5, 6), 'II.': (10, 6),

    # Источники
    'N': (-16, 7), 'O': (-13.5, 8.5), 'J': (-8.5, 8), 'I': (-7, 4.5),
    'E': (-5.5, 2.5), 'D': (-2, 2.5), 'H': (2.5, 4.5), 'K': (3, 9.5),
    'L': (2.5, 11), 'F': (9, 8), 'G': (11, 8), 'M': (4.5, 3.5),
    'B': (7, -2.5), 'C': (4, -3), 'P': (9, 2.5), 'A': (13, 2.5),

    # ТРАНЗИТНЫЕ УЗЛЫ
    'a': (4.3, 6), 'b': (3.7, 6),
    'c': (2.3, 6.7), 'd': (1.7, 7.1),
    'x': (1.5, 10.5), 'y': (1.5, 11.5),
    'e': (-5.0, 6.0),
    'g': (14.5, 0), 'h': (11.5, 0), 'i': (10.5, 0),
    'j': (8.0, 0), 'k': (6.5, 0), 'l': (0.0, 0),

    'm': (-5.6, 0), 'n': (-6.2, 0), 'o': (-6.8, 0), 'p': (-7.4, 0), 'q': (-8.0, 0),
    'r': (-9.5, 2), 's': (-10.5, 4),
    't': (-10.5, 6), 'u': (-9.5, 6),

    # ПОТРЕБИТЕЛИ
    '31': (4.3, 7), '30': (3.7, 7),
    '34': (2.3, 5.5), '33': (1.7, 6.0),
    '38': (2.2, 11.8), '1': (1.0, 12.5), '2': (1.0, 13.5),
    '32': (-5.0, 7.5),

    '35': (-10.5, 7.5), '36': (-9.5, 7.5),

    '5': (14.5, -1.5), '6': (16.0, 0), '7': (16.0, -1.5),
    '4': (11.5, -1.5), '9': (10.5, -1.5),
    '11': (8.0, 1.5), '10': (6.5, 1.5),
    '8': (0.0, 1.5), '29': (-3.5, -1.5),

    '23': (-5.6, -1.5), '26': (-6.2, 1.5), '22': (-6.8, -1.5), '21': (-7.4, 1.5), '28': (-8.0, -1.5),
    '20': (-9.5, 3), '19': (-10.5, 5),

    # СДВИНУТ УЗЕЛ 18 (чтобы не перекрывал линию к 13)
    '17': (-12, 1.5), '18': (-12.5, -0.8), '13': (-14.5, 0),
    '14': (-14.5, -1.5), '25': (-15.5, -1.5),
    '24': (-11, -3),
    '3': (-10, 1.5),
    '12': (5.5, 1.5),
    '27': (-14, 4.5),

    '15': (-12.5, 7.5), '16': (-12.5, 8.5),

    '37': (-7, 9.5)
}


def normalize_node(val):
    if val is None: return ""
    return str(val).strip()


def load_network_data(df_req, df_cap):
    requests = {}
    nodes_set = set()

    for u, v in BASE_TOPOLOGY:
        nodes_set.update([u, v])

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
        except (ValueError, TypeError):
            continue

    UNLIMITED_CAP = total_requested + 10000.0
    final_capacities = {}
    for u, v in BASE_TOPOLOGY:
        final_capacities[(u, v)] = UNLIMITED_CAP

    # Предварительно создаем граф для поиска путей по скрытым узлам
    G_base = nx.DiGraph(BASE_TOPOLOGY)
    
    for _, row in df_cap.iterrows():
        try:
            u = normalize_node(row['начало'])
            v = normalize_node(row['окончание'])
            cap = float(str(row['Допустимая мощность']).replace(',', '.').replace(' ', ''))

            if not u or not v: continue

            # Если это прямое ребро в топологии - ставим как есть
            if (u, v) in G_base.edges:
                final_capacities[(u, v)] = min(final_capacities.get((u, v), float('inf')), cap)
            else:
                # Ищем путь между "честными" узлами через скрытые
                try:
                    path = nx.shortest_path(G_base, u, v)
                    for i in range(len(path) - 1):
                        edge = (path[i], path[i+1])
                        # Ограничение распространяется на все сегменты пути
                        final_capacities[edge] = min(final_capacities.get(edge, float('inf')), cap)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    # Если пути нет в базе, но оно есть в файле - 
                    # создаем виртуальное ребро (на случай кастомных топологий)
                    final_capacities[(u, v)] = cap
                    nodes_set.update([u, v])
        except (ValueError, TypeError):
            continue

    nodes = list(nodes_set)
    destinations = list(set(dst for src, dst in requests.keys()))

    adj = {u: [] for u in nodes}
    for (u, v) in final_capacities.keys():
        if u in adj:
            adj[u].append(v)
        else:
            adj[u] = [v]

    return nodes, destinations, adj, final_capacities, requests
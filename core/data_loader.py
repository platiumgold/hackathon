import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Any, Optional, Set, Union

# Константы топологии для удобства управления
BASE_TOPOLOGY: List[Tuple[str, str]] = [
    ('A', 'XVIII.'), ('B', 'X.'), ('C', 'X.'), ('D', 'XV.'), ('E', 'XIV.'),
    ('F', 'II.'), ('G', 'II.'), ('H', 'III.'), ('I', 'V.'), ('J', 'VIII.'),
    ('K', 'XVI.'),
    ('M', 'X.'), ('N', 'VII.'), ('O', 'VII.'), ('P', 'XI.'),

    # Нижняя магистраль
    ('XVIII.', 'g'), ('g', '5'), ('g', '6'), ('6', '7'),
    ('XVIII.', 'h'), ('h', '4'), ('h', 'i'), ('i', '9'), ('i', 'XI.'),
    ('XI.', 'j'), ('j', '11'), ('j', 'k'), ('k', '10'), ('k', 'X.'),
    ('X.', '12'), ('X.', 'IX.'),
    ('IX.', 'l'), ('l', '8'), ('l', 'XV.'),

    ('XIV.', '29'),
    ('XV.', 'XIV.'),

    # Цепочка XIV -> XII
    ('XIV.', 'm'), ('m', '23'),
    ('m', 'n'), ('n', '26'),
    ('n', 'o'), ('o', '22'),
    ('o', 'p'), ('p', '21'),
    ('o', 'p'), ('p', '21'),
    ('p', 'q'), ('q', '28'),
    ('q', 'XII.'),

    ('XII.', '3'), ('XII.', 'XIII.'),
    ('XIII.', '17'),
    ('XIII.', '13'), ('XIII.', '18'), ('XIII.', '24'),
    ('13', '14'), ('13', '25'),

    # Цепочка XII -> VI
    ('XII.', 'r'), ('r', '20'),
    ('r', 's'), ('s', '19'),
    ('s', 'VI.'),

    # Верхняя магистраль
    ('VII.', 'VI.'),
    ('VI.', '27'), ('VI.', '15'),
    ('15', '16'),

    # Цепочка из VI для 35 и 36 и выход на V.
    ('VI.', 't'), ('t', 'u'), ('u', 'V.'),
    ('t', '35'), ('u', '36'),

    ('V.', 'VIII.'), ('VIII.', '37'),

    # Соединение магистралей
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

# Географические координаты узлов для визуализации
TOPOLOGY_POS: Dict[str, Tuple[float, float]] = {
    'XIII.': (-11, 0), 'XII.': (-8.5, 0), 'XIV.': (-5, 0), 'XV.': (-2, 0),
    'IX.': (1.5, 0), 'X.': (4.5, 0), 'XI.': (9, 0), 'XVIII.': (13, 0),
    'VII.': (-14, 6), 'VI.': (-11.5, 6), 'V.': (-7, 6), 'VIII.': (-7, 7.5),
    'XVII.': (-3, 6), 'III.': (3, 6), 'IV.': (1, 7.5), 'XVI.': (1, 9.5),
    'I.': (5, 6), 'II.': (10, 6),
    'N': (-16, 7), 'O': (-13.5, 8.5), 'J': (-8.5, 8), 'I': (-7, 4.5),
    'E': (-5.5, 2.5), 'D': (-2, 2.5), 'H': (2.5, 4.5), 'K': (3, 9.5),
    'L': (2.5, 11), 'F': (9, 8), 'G': (11, 8), 'M': (4.5, 3.5),
    'B': (7, -2.5), 'C': (4, -3), 'P': (9, 2.5), 'A': (13, 2.5),
    'a': (4.3, 6), 'b': (3.7, 6), 'c': (2.3, 6.7), 'd': (1.7, 7.1),
    'x': (1.5, 10.5), 'y': (1.5, 11.5), 'e': (-5.0, 6.0),
    'g': (14.5, 0), 'h': (11.5, 0), 'i': (10.5, 0),
    'j': (8.0, 0), 'k': (6.5, 0), 'l': (0.0, 0),
    'm': (-5.6, 0), 'n': (-6.2, 0), 'o': (-6.8, 0), 'p': (-7.4, 0), 'q': (-8.0, 0),
    'r': (-9.5, 2), 's': (-10.5, 4), 't': (-10.5, 6), 'u': (-9.5, 6),
    '31': (4.3, 7), '30': (3.7, 7), '34': (2.3, 5.5), '33': (1.7, 6.0),
    '38': (2.2, 11.8), '1': (1.0, 12.5), '2': (1.0, 13.5), '32': (-5.0, 7.5),
    '35': (-10.5, 7.5), '36': (-9.5, 7.5),
    '5': (14.5, -1.5), '6': (16.0, 0), '7': (16.0, -1.5),
    '4': (11.5, -1.5), '9': (10.5, -1.5), '11': (8.0, 1.5), '10': (6.5, 1.5),
    '8': (0.0, 1.5), '29': (-3.5, -1.5),
    '23': (-5.6, -1.5), '26': (-6.2, 1.5), '22': (-6.8, -1.5), '21': (-7.4, 1.5), '28': (-8.0, -1.5),
    '20': (-12.5, 2.5), '19': (-12.5, 3.5),
    '17': (-12, 1.5), '18': (-12.5, 0.5), '13': (-14.5, 0),
    '14': (-14.5, -1.5), '25': (-15.5, -1.5), '24': (-11, -1.5),
    '3': (-10, 1.5), '12': (5.5, 1.5), '27': (-14, 4.5),
    '15': (-12.5, 7.5), '16': (-12.5, 8.5), '37': (-7, 9.5)
}


def is_hidden(node: Any) -> bool:
    """Определяет, является ли узел 'скрытым' техническим узлом (строчные буквы)."""
    s = str(node)
    return s.islower() and s.isalpha()


def get_logical_edges() -> List[Tuple[str, str]]:
    """
    Возвращает список логических связей между реальными объектами,
    абстрагируясь от скрытых промежуточных узлов.
    """
    visible_nodes = [n for n in TOPOLOGY_POS.keys() if not is_hidden(n)]
    G = nx.DiGraph(BASE_TOPOLOGY)

    logical_edges = []
    for u in visible_nodes:
        for v in visible_nodes:
            if u == v: continue
            try:
                path = nx.shortest_path(G, u, v)
                # Логическая связь существует, если путь состоит только из скрытых узлов посередине
                if len(path) > 1 and all(is_hidden(node) for node in path[1:-1]):
                    logical_edges.append((u, v))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
    return sorted(logical_edges)


def get_current_topology(reversed_logical_edges: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Формирует список ребер с учетом развернутых направлений на пульте управления.
    """
    G = nx.DiGraph(BASE_TOPOLOGY)
    edges_to_remove = set()
    edges_to_add = set()

    for u, v in reversed_logical_edges:
        try:
            path = nx.shortest_path(G, u, v)
            for i in range(len(path) - 1):
                seg = (path[i], path[i + 1])
                edges_to_remove.add(seg)
                edges_to_add.add((path[i + 1], path[i]))
        except: continue

    new_edges = [e for e in BASE_TOPOLOGY if e not in edges_to_remove]
    new_edges.extend(list(edges_to_add))
    return new_edges


def normalize_node(val: Any) -> str:
    """Очистка и нормализация имени узла."""
    if val is None: return ""
    return str(val).strip()


def load_network_data(
    df_req: Optional[pd.DataFrame] = None, 
    df_cap: Optional[pd.DataFrame] = None, 
    topology_override: Optional[List[Tuple[str, str]]] = None
) -> Tuple[List[str], List[str], Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], List[Dict]]:
    """
    Загружает и валидирует все данные сети: топологию, заявки и ограничения.
    
    Returns:
        Tuple: (nodes, destinations, adj, capacities, requests, invalid_requests)
    """
    requests = {}
    invalid_requests = []
    nodes_set = set()

    active_topology = topology_override if topology_override is not None else BASE_TOPOLOGY
    for u, v in active_topology: nodes_set.update([u, v])

    G_active = nx.DiGraph(active_topology)
    G_undirected = nx.Graph(BASE_TOPOLOGY)

    total_requested = 0
    if df_req is not None:
        for _, row in df_req.iterrows():
            src, dst = normalize_node(row.get('Источник потока')), normalize_node(row.get('Потребитель'))
            try:
                val_str = str(row.get('Поток, кВт', 0)).replace(',', '.').replace(' ', '')
                vol = float(val_str)
                if src and dst and vol > 0:
                    nodes_set.update([src, dst])
                    if src in G_active and dst in G_active and nx.has_path(G_active, src, dst):
                        requests[(src, dst)] = requests.get((src, dst), 0.0) + vol
                        total_requested += vol
                    else:
                        invalid_requests.append({"Источник": src, "Потребитель": dst, "Объем (кВт)": vol})
            except (ValueError, TypeError): continue

    # Базовая пропускная способность для нелимитированных участков
    unlimited_cap = total_requested + 10000.0
    final_capacities = {(u, v): unlimited_cap for u, v in active_topology}

    if df_cap is not None:
        for _, row in df_cap.iterrows():
            try:
                u, v = normalize_node(row.get('начало')), normalize_node(row.get('окончание'))
                cap_val = row.get('Допустимая мощность', unlimited_cap)
                cap = float(str(cap_val).replace(',', '.').replace(' ', ''))
                if not u or not v: continue
                try:
                    path = nx.shortest_path(G_undirected, u, v)
                    for i in range(len(path) - 1):
                        n1, n2 = path[i], path[i + 1]
                        if (n1, n2) in final_capacities: final_capacities[(n1, n2)] = min(final_capacities[(n1, n2)], cap)
                        if (n2, n1) in final_capacities: final_capacities[(n2, n1)] = min(final_capacities[(n2, n1)], cap)
                except (nx.NetworkXNoPath, nx.NodeNotFound): pass
            except (ValueError, TypeError): continue

    nodes = list(nodes_set)
    destinations = list(set(dst for src, dst in requests.keys()))
    adj = {u: [] for u in nodes}
    for (u, v) in final_capacities.keys():
        if u in adj: adj[u].append(v)
    
    return nodes, destinations, adj, final_capacities, requests, invalid_requests
import random
import pandas as pd
import networkx as nx
from typing import Tuple, List, Dict, Any, Optional
from core.data_loader import BASE_TOPOLOGY, TOPOLOGY_POS


def is_hidden(node: Any) -> bool:
    """Проверяет, является ли узел техническим (строчные буквы)."""
    s = str(node)
    return s.islower() and s.isalpha()


def generate_synthetic_network(
    num_reqs: int = 15, 
    load_level: float = 100.0, 
    bottleneck_level: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Генерирует синтетические данные (заявки и ограничения) на основе топологии Альфа.
    
    Args:
        num_reqs: Количество генерируемых заявок.
        load_level: Средняя мощность одной заявки (кВт).
        bottleneck_level: Уровень дефицита (0.0 - избыток мощности, 1.0 - сильный дефицит).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (Датафрейм заявок, Датафрейм ограничений).
    """

    # 1. Классификация узлов
    all_nodes = list(TOPOLOGY_POS.keys())
    # Источники - только одиночные заглавные буквы
    sources = [n for n in all_nodes if str(n).isalpha() and len(str(n)) == 1 and str(n).isupper()]
    # Потребители - числовые обозначения
    consumers = [n for n in all_nodes if str(n).isdigit()]

    G = nx.DiGraph()
    G.add_nodes_from(TOPOLOGY_POS.keys())
    G.add_edges_from(BASE_TOPOLOGY)

    # 2. Генерация заявок
    req_data = []
    attempts = 0
    max_attempts = num_reqs * 20

    while len(req_data) < num_reqs and attempts < max_attempts:
        attempts += 1
        src, dst = random.choice(sources), random.choice(consumers)

        if nx.has_path(G, src, dst):
            # Вариация потока +/- 50%
            flow = round(load_level * random.uniform(0.5, 1.5), 1)
            if not any(r[0] == src and r[1] == dst for r in req_data):
                req_data.append([src, dst, flow])

    df_req = pd.DataFrame(req_data, columns=['Источник потока', 'Потребитель', 'Поток, кВт'])

    # 3. Генерация ограничений на логических участках
    visible_nodes = [n for n in TOPOLOGY_POS.keys() if not is_hidden(n)]
    logical_edges = []
    for u in visible_nodes:
        for v in visible_nodes:
            if u == v: continue
            try:
                path = nx.shortest_path(G, u, v)
                if len(path) > 1 and all(is_hidden(node) for node in path[1:-1]):
                    logical_edges.append((u, v))
            except (nx.NetworkXNoPath, nx.NodeNotFound): continue

    cap_data = []
    total_flow = df_req['Поток, кВт'].sum() if not df_req.empty else 1000.0

    # Выбираем количество участков с лимитами в зависимости от bottleneck_level
    num_limited = int(len(logical_edges) * (0.2 + 0.6 * bottleneck_level))
    if num_limited > 0:
        selected_logical = random.sample(logical_edges, min(num_limited, len(logical_edges)))
        for u, v in selected_logical:
            min_cap = load_level * 0.5
            # Чем выше дефицит, тем ниже потолок мощности
            max_cap = total_flow * (1.1 - bottleneck_level)
            cap = round(random.uniform(min_cap, max_cap), 1)
            cap_data.append([u, v, cap])

    df_cap = pd.DataFrame(cap_data, columns=['начало', 'окончание', 'Допустимая мощность'])

    return df_req, df_cap


if __name__ == "__main__":
    req, cap = generate_synthetic_network()
    print(f"Generated {len(req)} requests and {len(cap)} constraints.")
    req.to_csv("table_1_1_generated.csv", index=False)
    cap.to_csv("table_1_2_generated.csv", index=False)
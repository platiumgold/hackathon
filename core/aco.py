import numpy as np
import random
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional, Set

def run_aco(
    nodes: List[str], 
    destinations: List[str], 
    adj: Dict[str, List[str]], 
    capacities: Dict[Tuple[str, str], float], 
    requests: Dict[Tuple[str, str], float], 
    quantum: float = 10.0, 
    n_iterations: int = 30, 
    time_limit_min: int = 5, 
    early_stop: bool = True, 
    seed: int = 42
) -> Dict[str, Any]:
    """
    Запускает алгоритм муравьиной колонии (ACO) для оптимизации потоков в электросети.
    
    Алгоритм ищет оптимальные пути для каждой заявки, учитывая феромоны на ребрах
    и физические ограничения пропускной способности. При перегрузках используется
    механизм пропорционального ограничения (Fairness/Water-filling).

    Args:
        nodes: Список всех узлов сети.
        destinations: Список узлов-потребителей.
        adj: Словарь смежности (топология графа).
        capacities: Лимиты пропускной способности ребер {(u, v): capacity}.
        requests: Заявки на поставку {(src, dst): volume}.
        quantum: "Вес" одного муравья в кВт.
        n_iterations: Максимальное количество итераций.
        time_limit_min: Лимит времени выполнения в минутах.
        early_stop: Включить ли раннюю остановку при отсутствии прогресса.
        seed: Зерно генератора случайных чисел для воспроизводимости.

    Returns:
        Dict: Словарь с результатами последней итерации:
            - 'load_distribution': Нагрузка на каждое ребро.
            - 'delivered': Реально доставленные объемы по каждой заявке.
            - 'request_flows': Распределение потоков каждой заявки по ребрам.
            - 'successful_paths': Список найденных путей.
    """
    rng = np.random.RandomState(seed)
    DEFAULT_CAP = 0.0
    start_time = time.time()

    # Инициализация феромонов: (откуда) -> (куда) -> (цель) -> уровень
    pheromones = {
        u: {v: {dest: 1.0 for dest in destinations} for v in neighbors}
        for u, neighbors in adj.items()
    }

    history = []
    best_delivered = -1
    no_improve_iters = 0

    for iteration in range(n_iterations):
        # Проверка лимита времени (важно для регламента конкурса)
        if time.time() - start_time > time_limit_min * 60:
            print(f"ACO: Остановка по лимиту времени на итерации {iteration}")
            break

        paths_found = {req: [] for req in requests.keys()}

        # Фаза поиска путей муравьями
        for (src, dst), volume in requests.items():
            num_ants = max(10, min(int(volume / quantum) + 1, 150))
            max_steps = len(nodes) * 3

            for _ in range(num_ants):
                path = [src]
                visited = {src}
                step_count = 0

                while path and path[-1] != dst and step_count < max_steps:
                    step_count += 1
                    current_node = path[-1]
                    neighbors = adj.get(current_node, [])

                    valid_neighbors = [v for v in neighbors if v not in visited]
                    if not valid_neighbors:
                        path.pop()
                        continue

                    attractions = []
                    valid_choices = []
                    for v in valid_neighbors:
                        edge = (current_node, v)
                        cap = capacities.get(edge, DEFAULT_CAP)
                        if cap <= 0:
                            continue

                        # Вероятность выбора: Феромоны (память) * Пропускная способность (эвристика)
                        attr = (pheromones[current_node][v][dst] ** 1.0) * (cap ** 0.5)
                        attractions.append(attr)
                        valid_choices.append(v)

                    sum_attr = sum(attractions)
                    if sum_attr == 0.0:
                        path.pop()
                        continue

                    probs = [attr / sum_attr for attr in attractions]
                    next_node = rng.choice(valid_choices, p=probs)

                    path.append(next_node)
                    visited.add(next_node)

                if path and path[-1] == dst:
                    paths_found[(src, dst)].append(path)

        # Фаза распределения потоков и соблюдения ограничений
        current_load = {edge: 0.0 for edge in capacities.keys()}
        delivered = {req: 0.0 for req in requests.keys()}
        request_flows = {req: {} for req in requests.keys()}
        successful_paths = []

        all_path_allocs = []

        for req, volume in requests.items():
            paths = paths_found[req]
            if not paths: continue

            # Группируем муравьев по уникальным путям
            unique_paths = []
            path_counts = {}
            for p in paths:
                tp = tuple(p)
                if tp not in path_counts:
                    unique_paths.append(p)
                    path_counts[tp] = 0
                path_counts[tp] += 1

            total_ants = len(paths)
            for p in unique_paths:
                alloc_vol = volume * (path_counts[tuple(p)] / total_ants)
                path_obj = {'req': req, 'path': p, 'vol': alloc_vol, 'orig_vol': volume, 'current_vol': alloc_vol}
                all_path_allocs.append(path_obj)

        # Итеративное ограничение потоков (Water-filling logic)
        for _ in range(15):
            edge_loads = defaultdict(float)
            edge_req_flows = defaultdict(lambda: defaultdict(float))
            edge_req_paths = defaultdict(lambda: defaultdict(list))

            for po in all_path_allocs:
                if po['current_vol'] <= 0: continue
                p = po['path']
                req = po['req']
                for i in range(len(p) - 1):
                    edge = (p[i], p[i+1])
                    edge_loads[edge] += po['current_vol']
                    edge_req_flows[edge][req] += po['current_vol']
                    edge_req_paths[edge][req].append(po)

            overloaded = False
            path_scaling = {id(po): 1.0 for po in all_path_allocs}

            for edge, load in edge_loads.items():
                cap = capacities.get(edge, 0.0)
                if cap <= 0:
                    for req, pos in edge_req_paths[edge].items():
                        for po in pos:
                            path_scaling[id(po)] = 0.0
                    overloaded = True
                    continue

                if load <= cap + 1e-4:
                    continue

                overloaded = True
                req_flow_on_edge = edge_req_flows[edge]
                req_paths_on_edge = edge_req_paths[edge]

                # Пропорциональное распределение остатка мощности
                remaining_cap = cap
                uncapped = set(req_flow_on_edge.keys())
                final_alloc = {}

                while uncapped and remaining_cap > 1e-6:
                    total_orig_demand_uncapped = sum(requests[req] for req in uncapped)
                    if total_orig_demand_uncapped <= 0:
                        break

                    fitted = set()
                    for req in uncapped:
                        share = remaining_cap * (requests[req] / total_orig_demand_uncapped)
                        if req_flow_on_edge[req] <= share + 1e-6:
                            final_alloc[req] = req_flow_on_edge[req]
                            fitted.add(req)

                    if not fitted:
                        for req in uncapped:
                            final_alloc[req] = remaining_cap * (requests[req] / total_orig_demand_uncapped)
                        break

                    for req in fitted:
                        remaining_cap -= final_alloc[req]
                        uncapped.remove(req)

                for req, alloc in final_alloc.items():
                    actual = req_flow_on_edge[req]
                    if actual > alloc + 1e-6:
                        local_scale = alloc / actual
                        for po in req_paths_on_edge[req]:
                            path_scaling[id(po)] = min(path_scaling[id(po)], local_scale)

            for po in all_path_allocs:
                po['current_vol'] *= path_scaling[id(po)]

            if not overloaded:
                break

        # Обновление феромонов (Испарение и Осаждение)
        for u in pheromones:
            for v in pheromones[u]:
                for dest in pheromones[u][v]:
                    pheromones[u][v][dest] = max(pheromones[u][v][dest] * 0.7, 0.0001)

        for po in all_path_allocs:
            final_vol = po['current_vol']
            if final_vol > 1e-4:
                p = po['path']
                dst = po['req'][1]
                path_length = len(p) - 1
                if path_length > 0:
                    total_ants_for_req = len(paths_found[po['req']])
                    effective_ants = (final_vol / po['orig_vol']) * total_ants_for_req
                    reward = effective_ants * (10.0 / path_length)
                    for i in range(path_length):
                        u, v = p[i], p[i+1]
                        pheromones[u][v][dst] += reward

        # Собираем статистику итерации
        iter_delivered_sum = 0.0
        for po in all_path_allocs:
            final_vol = po['current_vol']
            if final_vol > 0.001:
                p = po['path']
                delivered[po['req']] += final_vol
                iter_delivered_sum += final_vol
                successful_paths.append((p, p[0], p[-1]))
                for i in range(len(p) - 1):
                    edge = (p[i], p[i+1])
                    if edge in current_load:
                        current_load[edge] += final_vol
                        request_flows[po['req']][edge] = request_flows[po['req']].get(edge, 0.0) + final_vol

        history.append({
            'load_distribution': current_load.copy(),
            'successful_paths': successful_paths,
            'delivered': delivered,
            'request_flows': request_flows
        })

        # Логика ранней остановки (Early Stopping)
        if early_stop:
            if iter_delivered_sum > best_delivered + 0.1:
                best_delivered = iter_delivered_sum
                no_improve_iters = 0
            else:
                no_improve_iters += 1
                if no_improve_iters >= max(5, n_iterations // 5):
                    print(f"ACO: Остановка по Early Stopping на {iteration} итерации")
                    break

    return history[-1] if history else {'load_distribution': {}, 'delivered': {}, 'request_flows': {}}
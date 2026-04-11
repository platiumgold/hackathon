import numpy as np
import random


def run_aco(nodes, destinations, adj, capacities, requests, quantum=10.0, n_iterations=30, seed=42):
    rng = np.random.RandomState(seed)
    DEFAULT_CAP = 0.0

    pheromones = {
        u: {v: {dest: 1.0 for dest in destinations} for v in neighbors}
        for u, neighbors in adj.items()
    }

    history = []
    for iteration in range(n_iterations):
        current_load = {edge: 0.0 for edge in capacities.keys()}
        successful_paths = []

        ants_pool = []
        for (src, dst), volume in requests.items():
            num_ants = int(volume / quantum)
            ants_pool.extend([(src, dst)] * num_ants)

        random.seed(seed + iteration)
        random.shuffle(ants_pool)

        for ant in ants_pool:
            current_node, target = ant
            start_node = current_node
            path = [current_node]
            visited = {current_node}

            while current_node != target:
                neighbors = adj.get(current_node, [])
                if not neighbors: break

                attractions, valid_neighbors = [], []
                for v in neighbors:
                    if v in visited: continue
                    edge = (current_node, v)
                    free_cap = capacities.get(edge, DEFAULT_CAP) - current_load.get(edge, 0.0)

                    if free_cap >= quantum:
                        attr = (pheromones[current_node][v][target] ** 1.0) * (min(free_cap, 1000.0) ** 3.0)
                        attractions.append(attr)
                        valid_neighbors.append(v)

                if not attractions: break

                sum_attr = sum(attractions)
                if sum_attr == 0.0: break

                probs = [attr / sum_attr for attr in attractions]
                next_node = rng.choice(valid_neighbors, p=probs)

                actual_edge = (current_node, next_node)
                current_load[actual_edge] = current_load.get(actual_edge, 0.0) + quantum

                path.append(next_node)
                visited.add(next_node)
                current_node = next_node

            if current_node == target:
                successful_paths.append((path, start_node, target))

        for u in pheromones:
            for v in pheromones[u]:
                for dest in pheromones[u][v]:
                    pheromones[u][v][dest] = max(pheromones[u][v][dest] * 0.7, 0.0001)

        for path, _, target in successful_paths:
            path_length = len(path) - 1
            for i in range(path_length):
                u, v = path[i], path[i + 1]
                pheromones[u][v][target] += (100.0 / path_length) * (capacities.get((u, v), 0.0) / quantum)

        history.append({'load_distribution': current_load.copy(), 'successful_paths': successful_paths})

    return history[-1]
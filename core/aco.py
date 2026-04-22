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

    # Pre-calculate node out-edges and capacities
    for iteration in range(n_iterations):
        # Phase 1: ACO path finding
        paths_found = {req: [] for req in requests.keys()}

        for (src, dst), volume in requests.items():
            num_ants = min(int(volume / quantum) + 1, 50) # Limit max ants per request for performance

            for _ in range(num_ants):
                current_node = src
                path = [current_node]
                visited = {current_node}

                while current_node != dst:
                    neighbors = adj.get(current_node, [])
                    if not neighbors: break

                    valid_neighbors = [v for v in neighbors if v not in visited]
                    if not valid_neighbors: break

                    attractions = []
                    for v in valid_neighbors:
                        edge = (current_node, v)
                        cap = capacities.get(edge, DEFAULT_CAP)
                        # Avoid 0 capacity edges
                        if cap <= 0:
                            attr = 0.0
                        else:
                            attr = (pheromones[current_node][v][dst] ** 1.0) * (cap ** 0.5)
                        attractions.append(attr)

                    sum_attr = sum(attractions)
                    if sum_attr == 0.0: break

                    probs = [attr / sum_attr for attr in attractions]
                    next_node = rng.choice(valid_neighbors, p=probs)

                    path.append(next_node)
                    visited.add(next_node)
                    current_node = next_node

                if current_node == dst:
                    paths_found[(src, dst)].append(path)

        # Update Pheromones
        for u in pheromones:
            for v in pheromones[u]:
                for dest in pheromones[u][v]:
                    pheromones[u][v][dest] = max(pheromones[u][v][dest] * 0.7, 0.0001)

        for req, paths in paths_found.items():
            dst = req[1]
            for path in paths:
                path_length = len(path) - 1
                for i in range(path_length):
                    u, v = path[i], path[i + 1]
                    pheromones[u][v][dst] += 10.0 / path_length

        # Phase 2: Flow allocation with proportional limits
        current_load = {edge: 0.0 for edge in capacities.keys()}
        delivered = {req: 0.0 for req in requests.keys()}
        request_flows = {req: {} for req in requests.keys()}
        successful_paths = []

        # Determine the flow request for each edge
        edge_requests = {edge: [] for edge in capacities.keys()}
        all_path_allocs = []

        for req, volume in requests.items():
            paths = paths_found[req]
            if not paths: continue

            # Count frequency of each unique path
            unique_paths = []
            path_counts = {}
            for p in paths:
                tp = tuple(p)
                if tp not in path_counts:
                    unique_paths.append(p)
                    path_counts[tp] = 0
                path_counts[tp] += 1

            # Distribute volume among found paths proportionally to ACO preference
            total_ants = len(paths)
            for p in unique_paths:
                alloc_vol = volume * (path_counts[tuple(p)] / total_ants)
                path_obj = {'req': req, 'path': p, 'vol': alloc_vol, 'orig_vol': volume}
                all_path_allocs.append(path_obj)

                for i in range(len(p) - 1):
                    edge = (p[i], p[i+1])
                    if edge in edge_requests:
                        edge_requests[edge].append(path_obj)

        # Iteratively scale down bottlenecks
        active_allocs = {id(po): po for po in all_path_allocs}
        path_scaling = {id(po): 1.0 for po in all_path_allocs}

        for edge, reqs_on_edge in edge_requests.items():
            cap = capacities.get(edge, 0.0)
            if cap <= 0:
                for po in reqs_on_edge:
                    path_scaling[id(po)] = 0.0
                continue

            total_req_vol = sum(po['vol'] for po in reqs_on_edge)
            if total_req_vol > cap:
                # Proportional limitation based on original requested volume
                scale_factor = cap / total_req_vol
                for po in reqs_on_edge:
                    path_scaling[id(po)] = min(path_scaling[id(po)], scale_factor)

        # Calculate final loads after scaling
        for po in all_path_allocs:
            final_vol = po['vol'] * path_scaling[id(po)]
            if final_vol > 0.001:
                p = po['path']
                delivered[po['req']] += final_vol
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

    return history[-1]
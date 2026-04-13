import numpy as np
import copy
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PPOAgent, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        probs = self.actor(state)
        value = self.critic(state)
        return probs, value

def run_rl(nodes, dests, adj, caps, reqs, epochs=1000, gamma=0.99, lr=3e-4, clip_ratio=0.2):
    """
    Solves the energy routing problem using Proximal Policy Optimization (PPO).
    """
    node_idx = {n: i for i, n in enumerate(nodes)}
    idx_node = {i: n for i, n in enumerate(nodes)}
    num_nodes = len(nodes)

    # Process edges
    edge_list = []
    edge_idx = {}
    idx_edge = {}
    for i, u in enumerate(nodes):
        for v in adj.get(u, []):
            if v in nodes:
                j = node_idx[v]
                edge_list.append((i, j))
                idx = len(edge_list) - 1
                edge_idx[(i, j)] = idx
                idx_edge[idx] = (i, j)
    num_edges = len(edge_list)

    # Processing requests
    req_list = []
    for (src, dst), amount in reqs.items():
        if src in nodes and dst in nodes:
            req_list.append((node_idx[src], node_idx[dst], amount))

    state_dim = num_edges # Current loads on edges
    # Action dimension: for each request, we want to pick a path.
    # To keep it simple, we generate a probability distribution over edges.
    # The action is a continuous vector of size num_edges, representing flow allocations.

    # We will use a simplified RL approach:
    # State: current edge remaining capacities
    action_dim = num_edges * len(req_list)

    # A simple actor-critic based on direct flow optimization
    # Actually, let's use a simpler heuristic with PPO:
    # We parameterize the policy as a distribution over edges for routing.

    actor_critic = PPOAgent(state_dim, action_dim)
    optimizer = optim.Adam(actor_critic.parameters(), lr=lr)

    # Normalize capacities
    max_cap = 1.0
    if caps:
        max_cap = max(caps.values())

    capacity_arr = np.zeros(num_edges)
    for (u, v), c in caps.items():
        if u in node_idx and v in node_idx:
            if (node_idx[u], node_idx[v]) in edge_idx:
                capacity_arr[edge_idx[(node_idx[u], node_idx[v])]] = c

    # Training loop
    best_load_distribution = {}
    best_delivered = {}
    best_reward = -float('inf')

    for epoch in range(epochs):
        state = np.copy(capacity_arr) / max_cap # Normalized remaining capacity
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        probs, value = actor_critic(state_tensor)

        # We interpret probs as weights for flow paths
        weights = probs.squeeze(0).reshape(len(req_list), num_edges)

        current_loads = np.zeros(num_edges)
        delivered = {req: 0.0 for req in range(len(req_list))}
        requested_flows = np.zeros((len(req_list), num_edges))

        reward = 0
        penalty = 0

        # Create copies of remaining amounts
        remaining = [amount for _, _, amount in req_list]
        initial_amounts = [amount for _, _, amount in req_list]

        # We process in small steps to approximate proportional fair sharing and allow multiple paths
        num_steps = 10

        for step in range(num_steps):
            step_requests = []
            for k in range(len(req_list)):
                if remaining[k] > 1e-5:
                    step_requests.append(k)

            if not step_requests:
                break

            # For each active request, find the shortest path based on current weights
            paths = {}
            for k in step_requests:
                s, d, _ = req_list[k]

                dist = {i: float('inf') for i in range(num_nodes)}
                prev = {i: None for i in range(num_nodes)}
                prev_edge = {i: None for i in range(num_nodes)}
                dist[s] = 0

                w_k = weights[k].detach().numpy() + 1e-6
                cost = -np.log(w_k)
                # Add penalty for congested edges to encourage alternative paths
                for e in range(num_edges):
                    if current_loads[e] >= capacity_arr[e] - 1e-5:
                        cost[e] += 1000.0 # Huge penalty if edge is full

                for _ in range(num_nodes - 1):
                    for e_idx, (u, v) in idx_edge.items():
                        if dist[u] + cost[e_idx] < dist[v]:
                            dist[v] = dist[u] + cost[e_idx]
                            prev[v] = u
                            prev_edge[v] = e_idx

                if dist[d] < 500.0: # Path found without using fully congested edges
                    curr = d
                    path_edges = []
                    valid = True
                    while curr != s:
                        e = prev_edge[curr]
                        if e is None:
                            valid = False
                            break
                        path_edges.append(e)
                        curr = prev[curr]

                    if valid and curr == s:
                        paths[k] = path_edges

            # Now we have paths for requests. We want to route a fraction of their remaining demand.
            # But we must respect edge capacities strictly and reduce proportionally if congested.

            # The maximum we try to route in this step for each request
            step_fractions = {k: remaining[k] / (num_steps - step) for k in paths}

            # Calculate desired load on each edge in this step
            desired_edge_loads = defaultdict(float)
            for k, edges in paths.items():
                for e in edges:
                    desired_edge_loads[e] += step_fractions[k]

            # Find the scaling factor for each request to not exceed any edge capacity
            scaling_factors = {k: 1.0 for k in paths}

            # Check for congestion and apply proportional reduction
            for e, desired in desired_edge_loads.items():
                available = max(0.0, capacity_arr[e] - current_loads[e])
                if desired > available:
                    # In case of no available capacity, reduce proportionally
                    # The task specifies proportional to initial amounts, but within this step,
                    # we reduce proportionally to their requested flow in this step (which traces back to remaining).
                    # Actually, let's strictly weight by initial_amounts
                    total_initial = sum(initial_amounts[k] for k, edges in paths.items() if e in edges)

                    for k, edges in paths.items():
                        if e in edges:
                            # Fair share for this request on this edge
                            fair_share = available * (initial_amounts[k] / total_initial if total_initial > 0 else 0)
                            # Maximum scaling factor we can allow for this request
                            allowed_factor = fair_share / step_fractions[k] if step_fractions[k] > 0 else 0
                            if allowed_factor < scaling_factors[k]:
                                scaling_factors[k] = allowed_factor

            # Apply the routed amounts
            for k, edges in paths.items():
                actual_route_amount = step_fractions[k] * scaling_factors[k]
                if actual_route_amount > 0:
                    for e in edges:
                        current_loads[e] += actual_route_amount
                        requested_flows[k, e] += actual_route_amount
                    delivered[k] += actual_route_amount
                    remaining[k] -= actual_route_amount
                    reward += actual_route_amount

        # Penalize overloads (although constrained logic prevents it, this helps if precision errors occur)
        overload = np.maximum(0, current_loads - capacity_arr)
        penalty = np.sum(overload) * 10
        total_reward = reward - penalty

        # Simplified policy gradient update
        # We want to increase the probability of edges used in requests that delivered flow
        # This is a very rough approximation of REINFORCE for this specific non-differentiable environment
        loss = 0
        for k in range(len(req_list)):
            if delivered[k] > 0:
                for e in range(num_edges):
                    if requested_flows[k, e] > 0:
                        # Maximize log prob of chosen actions
                        # Here, we treat 'weights' as probabilities
                        loss -= torch.log(weights[k, e] + 1e-8) * delivered[k]

        if type(loss) == torch.Tensor:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if total_reward > best_reward:
            best_reward = total_reward
            best_load_distribution = {}
            for e_idx in range(num_edges):
                if current_loads[e_idx] > 0:
                    u, v = idx_edge[e_idx]
                    best_load_distribution[(idx_node[u], idx_node[v])] = current_loads[e_idx]

            best_delivered = {}
            for k, (s, d, amount) in enumerate(req_list):
                if delivered[k] > 0:
                    best_delivered[(idx_node[s], idx_node[d])] = delivered[k]

    return {
        'load_distribution': best_load_distribution,
        'delivered': best_delivered
    }

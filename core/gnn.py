import torch
import torch.nn as nn
import torch.optim as optim


class PowerRoutingGNN(nn.Module):
    def __init__(self, num_nodes, hidden_dim=64):
        super().__init__()
        self.node_embeddings = nn.Embedding(num_nodes, hidden_dim)
        self.gcn = nn.Linear(hidden_dim, hidden_dim)
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128), nn.LeakyReLU(0.1),
            nn.Linear(128, 64), nn.LeakyReLU(0.1),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, adj_matrix, requests_list, node_to_idx, num_nodes):
        x = self.node_embeddings.weight
        x_combined = x + torch.relu(self.gcn(torch.matmul(adj_matrix, x)))

        flows = torch.zeros((len(requests_list), num_nodes, num_nodes))

        for req_idx, ((src, dst), demand) in enumerate(requests_list):
            src_idx, dst_idx = node_to_idx[src], node_to_idx[dst]
            max_flow = min(50.0, demand)  # Эмпирическое ограничение

            for u in range(num_nodes):
                for v in range(num_nodes):
                    if u != v:
                        ctx = torch.cat([x_combined[u], x_combined[v], x_combined[src_idx], x_combined[dst_idx]])
                        flows[req_idx, u, v] = self.edge_predictor(ctx) * max_flow
        return flows


def run_gnn(nodes, capacities, requests, epochs=300):
    num_nodes = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    requests_list = list(requests.items())

    cap_matrix = torch.zeros((num_nodes, num_nodes))
    for (u, v), cap in capacities.items():
        cap_matrix[node_to_idx[u], node_to_idx[v]] = cap

    adj_matrix = torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)
    model = PowerRoutingGNN(num_nodes)
    optimizer = optim.AdamW(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()
        flows = model(adj_matrix, requests_list, node_to_idx, num_nodes)

        total_edges = flows.sum(dim=0)
        loss_cap = torch.relu(total_edges - cap_matrix).sum()

        loss_cons, loss_dem = 0.0, 0.0
        for req_idx, ((src, dst), demand) in enumerate(requests_list):
            src_idx, dst_idx = node_to_idx[src], node_to_idx[dst]
            f_mat = flows[req_idx]
            in_f, out_f = f_mat.sum(dim=0), f_mat.sum(dim=1)

            for i in range(num_nodes):
                if i != src_idx and i != dst_idx:
                    loss_cons += torch.abs(in_f[i] - out_f[i])

            delivered = in_f[dst_idx] - out_f[dst_idx]
            loss_dem += ((torch.relu(demand - delivered) / demand) * 100.0) ** 2

        loss = (loss_cap * 10.0) + (loss_cons * 10.0) + (loss_dem * 1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    with torch.no_grad():
        final_flows = model(adj_matrix, requests_list, node_to_idx, num_nodes)

    return final_flows, node_to_idx, requests_list
import torch
import torch.nn as nn
import torch.optim as optim


class StrictEnergyGNN(nn.Module):
    def __init__(self, num_nodes, embed_dim=16):
        super().__init__()
        self.num_nodes = num_nodes
        # Эмбеддинги помогают сети понимать "соседство" узлов
        self.node_embed = nn.Embedding(num_nodes, embed_dim)

        # Оценка ребер (кто я -> куда иду -> конечная цель)
        self.edge_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Оценка поглощения (нужно ли оставить энергию здесь)
        self.sink_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def get_routing_matrices(self, adj_mask, dst_idx):
        """Возвращает вероятности перехода для конкретного узла назначения"""
        emb = self.node_embed.weight
        dst_emb = emb[dst_idx].expand(self.num_nodes, -1)

        # Считаем логиты для ребер
        edge_logits = torch.full((self.num_nodes, self.num_nodes), -1e9)
        for i in range(self.num_nodes):
            # Векторизуем соседей для i
            neighbors = torch.where(adj_mask[i] > 0)[0]
            if len(neighbors) > 0:
                e_in = torch.cat([
                    emb[i].expand(len(neighbors), -1),
                    emb[neighbors],
                    dst_emb[neighbors]
                ], dim=1)
                edge_logits[i, neighbors] = self.edge_mlp(e_in).squeeze(-1)

        # Логиты поглощения
        sink_input = torch.cat([emb, dst_emb], dim=1)
        sink_logits = self.sink_mlp(sink_input).squeeze(-1)

        # Softmax гарантирует, что сумма (выходные ребра + поглощение) = 1.0
        combined = torch.cat([edge_logits, sink_logits.unsqueeze(1)], dim=1)
        probs = torch.softmax(combined, dim=1)

        return probs[:, :self.num_nodes], probs[:, self.num_nodes]


def run_gnn(nodes, final_capacities, requests, epochs=250, lr=0.01):
    num_nodes = len(nodes)
    node_idx = {name: i for i, name in enumerate(nodes)}
    req_list = list(requests.items())
    num_reqs = len(req_list)

    # Подготовка масок
    adj_mask = torch.zeros((num_nodes, num_nodes))
    cap_matrix = torch.zeros((num_nodes, num_nodes))
    for (u, v), cap in final_capacities.items():
        if u in node_idx and v in node_idx:
            adj_mask[node_idx[u], node_idx[v]] = 1.0
            cap_matrix[node_idx[u], node_idx[v]] = float(cap)

    model = StrictEnergyGNN(num_nodes)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Глубина графа для симуляции (сколько шагов может пройти ток)
    max_steps = min(num_nodes, 10)

    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0
        all_req_flows = []

        for r_idx, ((src, dst), demand) in enumerate(req_list):
            s_i, d_i = node_idx[src], node_idx[dst]
            P_edge, P_sink = model.get_routing_matrices(adj_mask, d_i)

            current_dist = torch.zeros(num_nodes)
            current_dist[s_i] = float(demand)

            req_flow_matrix = torch.zeros((num_nodes, num_nodes))
            delivered = 0

            for _ in range(max_steps):
                step_flows = current_dist.unsqueeze(1) * P_edge
                req_flow_matrix += step_flows

                absorbed = current_dist * P_sink
                delivered += absorbed[d_i]

                # Штраф за потерю энергии (поглощение не в целевом узле)
                loss_waste = (absorbed.sum() - absorbed[d_i]) * 15.0
                total_loss += loss_waste

                current_dist = step_flows.sum(dim=0)

            # Штраф за недоставку
            total_loss += (float(demand) - delivered) * 20.0
            all_req_flows.append(req_flow_matrix)

        # Штраф за перегрузку ЛЭП
        total_flows = torch.stack(all_req_flows).sum(dim=0)
        overflow = torch.relu(total_flows - cap_matrix)
        total_loss += overflow.sum() * 10.0

        total_loss.backward()
        optimizer.step()

    # Финальный расчет результатов
    with torch.no_grad():
        final_results = torch.zeros((num_reqs, num_nodes, num_nodes))
        for r_idx, ((src, dst), demand) in enumerate(req_list):
            s_i, d_i = node_idx[src], node_idx[dst]
            P_edge, P_sink = model.get_routing_matrices(adj_mask, d_i)

            curr = torch.zeros(num_nodes)
            curr[s_i] = float(demand)
            for _ in range(max_steps):
                step_f = curr.unsqueeze(1) * P_edge
                final_results[r_idx] += step_f
                curr = step_f.sum(dim=0)

    return final_results, node_idx, req_list
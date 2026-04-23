import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from collections import defaultdict
import numpy as np


class StrictEnergyGNN(nn.Module):
    def __init__(self, num_nodes, embed_dim=32):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_embed = nn.Embedding(num_nodes, embed_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def get_edge_probabilities(self, adj_mask, dst_idx):
        """Считает вероятности перехода по ребрам на основе графовых признаков"""
        emb = self.node_embed.weight
        dst_emb = emb[dst_idx].expand(self.num_nodes, -1)

        edge_logits = torch.full((self.num_nodes, self.num_nodes), -1e9, device=emb.device)
        for i in range(self.num_nodes):
            neighbors = torch.where(adj_mask[i] > 0)[0]
            if len(neighbors) > 0:
                e_in = torch.cat([
                    emb[i].expand(len(neighbors), -1),
                    emb[neighbors],
                    dst_emb[neighbors]
                ], dim=1)
                edge_logits[i, neighbors] = self.edge_mlp(e_in).squeeze(-1)

        # Нормализуем вероятности для каждого узла
        probs = torch.softmax(edge_logits, dim=1)
        # Убираем вероятности там, где нет физических связей
        probs = probs * adj_mask
        return probs


def run_gnn(nodes, final_capacities, requests, epochs=250, lr=0.005):
    num_nodes = len(nodes)
    node_idx = {name: i for i, name in enumerate(nodes)}
    reverse_node_idx = {i: name for i, name in enumerate(nodes)}

    req_list = []
    for (src, dst), amount in requests.items():
        if src in node_idx and dst in node_idx:
            req_list.append(((src, dst), amount))

    if not req_list:
        return {'load_distribution': {}, 'delivered': {}, 'request_flows': {}}

    adj_mask = torch.zeros((num_nodes, num_nodes))
    cap_matrix = torch.zeros((num_nodes, num_nodes))
    for (u, v), cap in final_capacities.items():
        if u in node_idx and v in node_idx:
            adj_mask[node_idx[u], node_idx[v]] = 1.0
            cap_matrix[node_idx[u], node_idx[v]] = float(cap)

    model = StrictEnergyGNN(num_nodes, embed_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- ЭТАП 1: Обучение GNN для понимания структуры графа ---
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0.0

        for r_idx, ((src, dst), demand) in enumerate(req_list):
            s_i, d_i = node_idx[src], node_idx[dst]
            probs = model.get_edge_probabilities(adj_mask, d_i)

            # Штраф: направлять потоки в сторону цели
            # Используем простую эвристику расстояний
            path_loss = -torch.log(probs[s_i, :].clamp(min=1e-5)).mean()
            total_loss += path_loss

        if total_loss > 0:
            total_loss.backward()
            optimizer.step()

    # --- ЭТАП 2: Применение бизнес-логики (Evaluation & Routing) ---
    model.eval()

    delivered_dict = {(src, dst): 0.0 for (src, dst), _ in req_list}
    load_distribution = defaultdict(float)
    request_flows_dict = defaultdict(lambda: defaultdict(float))

    # Текущие свободные мощности в сети (динамически обновляются)
    current_capacities = {k: v for k, v in final_capacities.items()}

    # Чтобы соблюсти условие пропорциональности, бьем объемы на итерации (например 50 шагов)
    # На каждом шаге каждая заявка пытается передать 2% от своего изначального объема
    num_steps = 50
    step_demands = {req_key: amount / num_steps for req_key, amount in req_list}

    with torch.no_grad():
        # Предрассчитаем веса ребер (штрафы) для каждой заявки на основе GNN
        gnn_weights = {}
        for (src, dst), _ in req_list:
            d_i = node_idx[dst]
            probs = model.get_edge_probabilities(adj_mask, d_i).numpy()
            # Чем выше вероятность GNN, тем ниже вес (сопротивление) ребра
            weight_matrix = -np.log(probs + 1e-9)
            gnn_weights[(src, dst)] = weight_matrix

    for step in range(num_steps):
        for (src, dst), amount in req_list:
            req_key = (src, dst)
            amount_to_send = step_demands[req_key]

            if amount_to_send <= 0:
                continue

            # Ищем пути, пока не передадим порцию энергии или пока не закончатся пути
            while amount_to_send > 1e-4:
                # 1. Строим граф только из доступных ребер
                G = nx.DiGraph()
                for (u, v), cap in current_capacities.items():
                    if cap > 1e-4:  # Ребро доступно
                        u_idx, v_idx = node_idx[u], node_idx[v]
                        # Вес ребра из GNN
                        w = gnn_weights[req_key][u_idx, v_idx]
                        G.add_edge(u, v, weight=w)

                # 2. Ищем путь
                try:
                    # GNN направляет поиск кратчайшего пути
                    path = nx.shortest_path(G, source=src, target=dst, weight='weight')
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    # Если путей больше нет, значит лимит по всем направлениям исчерпан
                    # Оставшаяся энергия просто не доставляется (срабатывает пропорциональное ограничение)
                    break

                    # 3. Находим "узкое горлышко" на найденном пути
                path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                bottleneck_cap = min(current_capacities[e] for e in path_edges)

                # Сколько энергии реально можем протолкнуть по этому пути прямо сейчас
                flow_to_push = min(amount_to_send, bottleneck_cap)

                # 4. Проталкиваем энергию и обновляем состояние
                delivered_dict[req_key] += flow_to_push
                amount_to_send -= flow_to_push

                for e in path_edges:
                    current_capacities[e] -= flow_to_push
                    load_distribution[e] += flow_to_push
                    request_flows_dict[req_key][e] += flow_to_push

    # Очистка от пустых значений для красоты словаря
    clean_request_flows = {
        req: dict(flows) for req, flows in request_flows_dict.items() if flows
    }

    return {
        'load_distribution': dict(load_distribution),
        'delivered': delivered_dict,
        'request_flows': clean_request_flows
    }
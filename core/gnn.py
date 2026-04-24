import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from collections import defaultdict
import numpy as np
import time
from typing import List, Dict, Tuple, Any, Optional, Set, Union


class StrictEnergyGNN(nn.Module):
    """
    Графовая нейронная сеть (GNN) для оценки вероятностей перетоков.
    
    Использует эмбеддинги узлов и MLP для предсказания привлекательности ребер
    в зависимости от топологии и целевого узла (потребителя).
    """
    def __init__(self, num_nodes: int, embed_dim: int = 32):
        """
        Args:
            num_nodes: Общее количество узлов в сети.
            embed_dim: Размерность вектора признаков (эмбеддинга) узла.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.node_embed = nn.Embedding(num_nodes, embed_dim)

        # Слой для вычисления весов ребер на основе признаков (Source, Target, Destination)
        self.edge_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def get_edge_probabilities(self, adj_mask: torch.Tensor, dst_idx: int) -> torch.Tensor:
        """
        Вычисляет вероятности перехода между узлами.
        
        Args:
            adj_mask: Матрица смежности (1.0 если связь есть, иначе 0.0).
            dst_idx: Индекс узла-потребителя (цель потока).
            
        Returns:
            torch.Tensor: Матрица вероятностей [N x N].
        """
        emb = self.node_embed.weight
        dst_emb = emb[dst_idx].expand(self.num_nodes, -1)

        edge_logits = torch.full((self.num_nodes, self.num_nodes), -1e9, device=emb.device)
        for i in range(self.num_nodes):
            neighbors = torch.where(adj_mask[i] > 0)[0]
            if len(neighbors) > 0:
                # Вход: признаки текущего узла, соседа и финальной цели
                e_in = torch.cat([
                    emb[i].expand(len(neighbors), -1),
                    emb[neighbors],
                    dst_emb[neighbors]
                ], dim=1)
                edge_logits[i, neighbors] = self.edge_mlp(e_in).squeeze(-1)

        # Softmax для получения вероятностей (сумма по строке = 1.0)
        probs = torch.softmax(edge_logits, dim=1)
        # Маскирование (только физически существующие ребра)
        probs = probs * adj_mask
        return probs


def run_gnn(
    nodes: List[str], 
    final_capacities: Dict[Tuple[str, str], float], 
    requests: Dict[Tuple[str, str], float], 
    epochs: int = 50, 
    time_limit_min: int = 5, 
    early_stop: bool = True, 
    lr: float = 0.005
) -> Dict[str, Any]:
    """
    Запускает Physics-Informed GNN для решения задачи маршрутизации.
    
    Алгоритм состоит из двух этапов:
    1. Обучение GNN понимать структуру графа и цели потоков.
    2. Пошаговое распределение энергии (Incremental Routing) на основе весов от GNN.
    
    Args:
        nodes: Список узлов.
        final_capacities: Лимиты ребер.
        requests: Заявки.
        epochs: Число эпох обучения.
        time_limit_min: Лимит времени (мин).
        early_stop: Включить ли раннюю остановку.
        lr: Скорость обучения (Learning Rate).
    """
    start_time = time.time()
    num_nodes = len(nodes)
    node_idx = {name: i for i, name in enumerate(nodes)}
    
    req_list = []
    for (src, dst), amount in requests.items():
        if src in node_idx and dst in node_idx:
            req_list.append(((src, dst), amount))

    if not req_list:
        return {'load_distribution': {}, 'delivered': {}, 'request_flows': {}}

    adj_mask = torch.zeros((num_nodes, num_nodes))
    for (u, v), _ in final_capacities.items():
        if u in node_idx and v in node_idx:
            adj_mask[node_idx[u], node_idx[v]] = 1.0

    model = StrictEnergyGNN(num_nodes, embed_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    patience_counter = 0

    # ЭТАП 1: Обучение (Поиск графовых закономерностей)
    model.train()
    for epoch in range(epochs):
        if time.time() - start_time > time_limit_min * 60:
            break

        optimizer.zero_grad()
        total_loss = 0.0

        for _, ((src, dst), _) in enumerate(req_list):
            s_i, d_i = node_idx[src], node_idx[dst]
            probs = model.get_edge_probabilities(adj_mask, d_i)
            # Минимизируем отрицательный логарифм вероятности выбора соседа в сторону цели
            path_loss = -torch.log(probs[s_i, :].clamp(min=1e-5)).mean()
            total_loss += path_loss

        if total_loss > 0:
            total_loss.backward()
            optimizer.step()

        if early_stop:
            loss_val = total_loss.item()
            if loss_val < best_loss - 0.001:
                best_loss, patience_counter = loss_val, 0
            else:
                patience_counter += 1
                if patience_counter > 15: break

    # ЭТАП 2: Инкрементальная маршрутизация (бизнес-логика)
    model.eval()
    delivered_dict = {(src, dst): 0.0 for (src, dst), _ in req_list}
    load_distribution = defaultdict(float)
    request_flows_dict = defaultdict(lambda: defaultdict(float))
    current_caps = {k: v for k, v in final_capacities.items()}

    # Разбиваем на мелкие порции для обеспечения пропорциональности (Fairness)
    num_steps = 50
    step_demands = {req_key: amount / num_steps for req_key, amount in req_list}

    with torch.no_grad():
        gnn_weights = {}
        for (src, dst), _ in req_list:
            probs = model.get_edge_probabilities(adj_mask, node_idx[dst]).numpy()
            gnn_weights[(src, dst)] = -np.log(probs + 1e-9)

    for _ in range(num_steps):
        for (src, dst), _ in req_list:
            req_key, amount_to_send = (src, dst), step_demands[(src, dst)]
            while amount_to_send > 1e-4:
                G = nx.DiGraph()
                for (u, v), cap in current_caps.items():
                    if cap > 1e-4:
                        G.add_edge(u, v, weight=gnn_weights[req_key][node_idx[u], node_idx[v]])
                try:
                    path = nx.shortest_path(G, source=src, target=dst, weight='weight')
                except (nx.NetworkXNoPath, nx.NodeNotFound): break

                path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                bottleneck = min(current_caps[e] for e in path_edges)
                flow = min(amount_to_send, bottleneck)

                delivered_dict[req_key] += flow
                amount_to_send -= flow
                for e in path_edges:
                    current_caps[e] -= flow
                    load_distribution[e] += flow
                    request_flows_dict[req_key][e] += flow

    return {
        'load_distribution': dict(load_distribution),
        'delivered': delivered_dict,
        'request_flows': {req: dict(flows) for req, flows in request_flows_dict.items() if flows}
    }
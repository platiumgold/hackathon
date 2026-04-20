import random

import torch
import torch.nn as nn
import torch.optim as optim


class PowerRoutingGNN(nn.Module):
    def __init__(self, num_nodes, hidden_dim=64):
        super().__init__()
        self.node_embeddings = nn.Embedding(num_nodes, hidden_dim)
        self.gcn = nn.Linear(hidden_dim, hidden_dim)
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, cap_matrix, requests_list, node_to_idx, edge_pairs):
        x = self.node_embeddings.weight
        # Небольшая "свертка" по структуре: используем только матрицу мощности как подсказку.
        # (Для скорости и детерминизма не строим полную матрицу смежности по всем u,v.)
        x_combined = x + torch.relu(self.gcn(torch.matmul((cap_matrix > 0).float(), x)))

        num_nodes = x_combined.shape[0]
        flows = torch.zeros(
            (len(requests_list), num_nodes, num_nodes),
            device=cap_matrix.device,
            dtype=x_combined.dtype,
        )

        for req_idx, ((src, dst), demand) in enumerate(requests_list):
            src_idx, dst_idx = node_to_idx[src], node_to_idx[dst]

            for u, v in edge_pairs:
                if u == v:
                    continue
                # Делаем destination (потребитель) тупиковым узлом:
                # для этой заявки не отправляем поток дальше от dst.
                if u == dst_idx:
                    continue
                # Не моделируем "петли" в сторону источника.
                if v == src_idx:
                    continue
                ctx = torch.cat(
                    [
                        x_combined[u],
                        x_combined[v],
                        x_combined[src_idx],
                        x_combined[dst_idx],
                    ]
                )
                edge_cap = cap_matrix[u, v]
                flows[req_idx, u, v] = self.edge_predictor(ctx) * edge_cap
        return flows


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _apply_proportional_edge_scaling(
    flows: torch.Tensor, cap_matrix: torch.Tensor, edge_pairs
) -> torch.Tensor:
    """
    Жестко приводит решение к допустимым мощностям по участкам.

    Если суммарная нагрузка на ребре превышает cap, все заявки,
    использующие это ребро, масштабируются пропорционально.
    """
    with torch.no_grad():
        total_edges = flows.sum(dim=0)  # [N, N]
        out = flows.clone()

        for u, v in edge_pairs:
            cap = cap_matrix[u, v].item()
            load = total_edges[u, v].item()
            if cap > 0 and load > cap + 1e-9:
                scale = cap / max(load, 1e-12)
                out[:, u, v] = out[:, u, v] * scale
        return out


def _hard_capacity_scale_torch(flows: torch.Tensor, cap_matrix: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Дифференцируемое жесткое масштабирование по мощностям участков.

    Для каждого ребра (u,v):
      - если суммарная нагрузка total_edges[u,v] > cap[u,v], то умножаем все потоки по этому ребру на cap/total_edges
      - иначе оставляем как есть
    """
    cap = cap_matrix
    total_edges = flows.sum(dim=0)  # [N, N]

    eps_t = torch.tensor(eps, device=flows.device, dtype=flows.dtype)

    cap_pos = cap > 0
    scale_raw = cap / (total_edges + eps_t)

    # scale = 1 если total_edges <= cap, иначе cap/total_edges
    ones = torch.ones_like(total_edges)
    scale = torch.where(total_edges > cap, scale_raw, ones)
    # на ребрах без ограничений мощности обнуляем (должно быть 0 и так, но на всякий случай)
    scale = torch.where(cap_pos, scale, torch.zeros_like(scale))
    return flows * scale.unsqueeze(0)


def run_gnn(
        nodes,
        capacities,
        requests,
        epochs=300,
        seed: int = 42,
        cap_penalty_weight: float = 1.0,
        proportionality_penalty_weight: float = 200.0,
):
    """
    Обучает GNN-модель маршрутизации с жестким учетом пропускных способностей.
    Возвращает финальное поле потоков, индекс узлов и список заявок.
    """
    _set_global_seed(seed)

    num_nodes = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    requests_list = list(requests.items())

    cap_matrix = torch.zeros((num_nodes, num_nodes))
    for (u, v), cap in capacities.items():
        cap_matrix[node_to_idx[u], node_to_idx[v]] = cap

    edge_indices = torch.nonzero(cap_matrix > 0, as_tuple=False)  # [E, 2]
    edge_pairs = [(int(i), int(j)) for i, j in edge_indices.tolist()]

    model = PowerRoutingGNN(num_nodes)
    optimizer = optim.AdamW(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()
        flows = model(cap_matrix, requests_list, node_to_idx, edge_pairs)
        flows = _hard_capacity_scale_torch(flows, cap_matrix)

        total_edges = flows.sum(dim=0)
        cap_excess = torch.relu(total_edges - cap_matrix)
        loss_cap = cap_excess.sum()

        loss_cons, loss_dem, loss_overdelivery = 0.0, 0.0, 0.0

        delivered_by_req = []
        demand_by_req = []

        for req_idx, ((src, dst), demand) in enumerate(requests_list):
            src_idx, dst_idx = node_to_idx[src], node_to_idx[dst]
            f_mat = flows[req_idx]
            in_f, out_f = f_mat.sum(dim=0), f_mat.sum(dim=1)

            mask = torch.ones(num_nodes, device=flows.device, dtype=torch.bool)
            mask[src_idx] = False
            mask[dst_idx] = False
            loss_cons = loss_cons + torch.abs((in_f - out_f)[mask]).sum()

            delivered = torch.relu(in_f[dst_idx] - out_f[dst_idx])
            demand_t = torch.tensor(float(demand), device=flows.device, dtype=flows.dtype).clamp_min(1e-9)

            # Штраф за недоставку
            loss_dem += (torch.abs(demand_t - delivered) / demand_t * 100.0) ** 2

            # ЖЕСТКИЙ ШТРАФ за перевыполнение (создание лишней энергии из воздуха)
            loss_overdelivery += torch.relu(delivered - demand_t) * 1000.0

            delivered_by_req.append(delivered)
            demand_by_req.append(demand_t)

        # ГЛОБАЛЬНЫЙ ПРОПОРЦИОНАЛЬНЫЙ ШТРАФ (Требование равномерности)
        # Все заявки должны удовлетворяться в равном процентном соотношении
        fractions = [deliv / dem for deliv, dem in zip(delivered_by_req, demand_by_req)]
        fractions_t = torch.stack(fractions)
        mean_frac = fractions_t.mean()
        # Штрафуем дисперсию: чем сильнее разброс процентов доставки, тем больше штраф
        loss_prop = torch.sum((fractions_t - mean_frac) ** 2)

        loss = (loss_cap * cap_penalty_weight) + (loss_cons * 10.0) + (loss_dem * 1.0) + \
               (loss_prop * proportionality_penalty_weight) + loss_overdelivery

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Пост-процессинг для жесткого обрезания
    with torch.no_grad():
        final_flows = model(cap_matrix, requests_list, node_to_idx, edge_pairs)
        final_flows = _hard_capacity_scale_torch(final_flows, cap_matrix)

        # Запрещаем доставлять больше, чем запрошено
        for req_idx, ((src, dst), demand) in enumerate(requests_list):
            src_idx, dst_idx = node_to_idx[src], node_to_idx[dst]
            in_f = final_flows[req_idx].sum(dim=0)
            out_f = final_flows[req_idx].sum(dim=1)
            delivered = (in_f[dst_idx] - out_f[dst_idx]).item()

            if delivered > demand:
                # Масштабируем всю матрицу потоков заявки вниз, чтобы доставка = demand
                scale = demand / max(delivered, 1e-9)
                final_flows[req_idx] = final_flows[req_idx] * scale

    return final_flows, node_to_idx, requests_list
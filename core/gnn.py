import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class PowerRoutingGNN(nn.Module):
    def __init__(self, num_nodes, hidden_dim=64):
        super().__init__()
        self.node_embeddings = nn.Embedding(num_nodes, hidden_dim)

        # Легкие графовые свертки для понимания соседей
        self.gcn1 = nn.Linear(hidden_dim, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, hidden_dim)

        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            # Выдаем строго долю от 0 до 1
            nn.Sigmoid()
        )

    def forward(self, cap_matrix, req_data, edge_pairs):
        x = self.node_embeddings.weight
        adj = (cap_matrix > 0).float()

        x = x + torch.relu(self.gcn1(torch.matmul(adj, x)))
        x_combined = x + torch.relu(self.gcn2(torch.matmul(adj, x)))

        R = len(req_data)
        N = x_combined.shape[0]
        E = len(edge_pairs)

        if E == 0 or R == 0:
            return torch.zeros((R, N, N), device=cap_matrix.device)

        u_idx = torch.tensor([u for u, v in edge_pairs], device=cap_matrix.device)
        v_idx = torch.tensor([v for u, v in edge_pairs], device=cap_matrix.device)
        src_idx = req_data[:, 0].long()
        dst_idx = req_data[:, 1].long()
        demands = req_data[:, 2]

        ctx = torch.cat([
            x_combined[u_idx].unsqueeze(0).expand(R, E, -1),
            x_combined[v_idx].unsqueeze(0).expand(R, E, -1),
            x_combined[src_idx].unsqueeze(1).expand(R, E, -1),
            x_combined[dst_idx].unsqueeze(1).expand(R, E, -1)
        ], dim=-1)

        # Сеть предсказывает ДОЛЮ (0.0 - 1.0), а не киловатты
        edge_fractions = self.edge_predictor(ctx).squeeze(-1)

        # Умножаем долю на реальный спрос, получая физические киловатты
        dem_t = demands.unsqueeze(1)
        flows_e = edge_fractions * dem_t

        # Запрет невозможных петель (в источник или из потребителя)
        mask_u_is_dst = u_idx.unsqueeze(0) == dst_idx.unsqueeze(1)
        mask_v_is_src = v_idx.unsqueeze(0) == src_idx.unsqueeze(1)
        flows_e = flows_e.masked_fill(mask_u_is_dst | mask_v_is_src, 0.0)

        flows = torch.zeros((R, N, N), device=cap_matrix.device)
        for r in range(R):
            flows[r, u_idx, v_idx] = flows_e[r]

        return flows


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _hard_capacity_scale_torch(flows: torch.Tensor, cap_matrix: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    total_edges = flows.sum(dim=0)
    eps_t = torch.tensor(eps, device=flows.device, dtype=flows.dtype)
    scale_raw = cap_matrix / (total_edges + eps_t)

    ones = torch.ones_like(total_edges)
    scale = torch.where(total_edges > cap_matrix, scale_raw, ones)
    scale = torch.where(cap_matrix > 0, scale, torch.zeros_like(scale))
    return flows * scale.unsqueeze(0)


def run_gnn(
        nodes, capacities, requests, epochs=500, seed: int = 42,
        cap_penalty_weight: float = 20.0,
        proportionality_penalty_weight: float = 10.0
):
    _set_global_seed(seed)

    num_nodes = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    requests_list = list(requests.items())

    cap_matrix = torch.zeros((num_nodes, num_nodes))
    for (u, v), cap in capacities.items():
        cap_matrix[node_to_idx[u], node_to_idx[v]] = cap

    edge_indices = torch.nonzero(cap_matrix > 0, as_tuple=False)
    edge_pairs = [(int(i), int(j)) for i, j in edge_indices.tolist()]

    req_data = torch.tensor([
        [node_to_idx[src], node_to_idx[dst], demand]
        for ((src, dst), demand) in requests_list
    ], dtype=torch.float32)

    model = PowerRoutingGNN(num_nodes)
    # Стабильный шаг обучения
    optimizer = optim.AdamW(model.parameters(), lr=0.01)

    print("\n" + "=" * 50)
    print("🚀 СТАРТ GNN (Нормализованная стабильная физика)")
    print("=" * 50)

    R = len(requests_list)
    src_idx = req_data[:, 0].long()
    dst_idx = req_data[:, 1].long()
    demands = req_data[:, 2]
    batch_idx = torch.arange(R)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Получаем сырые потоки
        flows = model(cap_matrix, req_data, edge_pairs)

        in_f = flows.sum(dim=1)
        out_f = flows.sum(dim=2)
        delivered = torch.relu(in_f[batch_idx, dst_idx] - out_f[batch_idx, dst_idx])

        # ========================================================
        # НОРМАЛИЗОВАННЫЕ ШТРАФЫ (Значения всегда в диапазоне ~0-10)
        # ========================================================

        # 1. Штраф доставки (Стремится к 0, когда доставлено = спрос)
        delivery_ratio = delivered / (demands + 1e-5)
        loss_dem = F.mse_loss(delivery_ratio, torch.ones_like(delivery_ratio)) * 10.0

        # 2. Штраф перегруза сети (В долях от вместимости трубы)
        total_edges = flows.sum(dim=0)
        cap_ratio = total_edges / (cap_matrix + 1e-5)
        # Штрафуем только то, что превышает 1.0 (т.е. 100% вместимости)
        overload = torch.relu(cap_ratio - 1.0)
        loss_cap = overload.mean() * cap_penalty_weight

        # 3. Штраф Кирхгофа (Входящий ток должен равняться исходящему)
        mask = torch.ones((R, num_nodes), dtype=torch.bool, device=flows.device)
        mask[batch_idx, src_idx] = False
        mask[batch_idx, dst_idx] = False

        # Нормализуем утечки относительно размера заявки
        demands_exp = demands.unsqueeze(1).expand(-1, num_nodes)[mask] + 1e-5
        transit_in = in_f[mask] / demands_exp
        transit_out = out_f[mask] / demands_exp
        loss_cons = F.mse_loss(transit_in, transit_out) * 5.0

        # 4. Штраф справедливости (Дисперсия долей доставки)
        loss_fair = torch.var(delivery_ratio) * proportionality_penalty_weight

        # Итоговый loss теперь состоит из маленьких чисел (например, 2.5 + 0.8 + 1.1)
        # Взрыв градиентов математически невозможен.
        loss = loss_dem + loss_cap + loss_cons + loss_fair
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(
                f"Эпоха {epoch:3d} | Заявки: {demands.sum().item():.0f} | Доставлено: {delivered.sum().item():.0f} кВт "
                f"| Loss Норм. Доставки: {loss_dem.item():.2f} | Loss Норм. Перегруза: {loss_cap.item():.2f}")

    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО\n")

    # ========================================================
    # ПОСТ-ПРОЦЕССИНГ (Жесткое приведение к физическим законам)
    # ========================================================
    with torch.no_grad():
        final_flows = model(cap_matrix, req_data, edge_pairs)

        # 1. Справедливое пропорциональное распределение дефицита в трубах
        final_flows = _hard_capacity_scale_torch(final_flows, cap_matrix)

        # 2. Гарантия, что ни одна заявка не получит больше, чем просила
        for req_idx in range(R):
            d = dst_idx[req_idx]
            dem = demands[req_idx].item()

            in_d = final_flows[req_idx].sum(dim=0)[d]
            out_d = final_flows[req_idx].sum(dim=1)[d]
            delivered_val = (in_d - out_d).item()

            if delivered_val > dem + 1e-4:
                scale = dem / max(delivered_val, 1e-9)
                final_flows[req_idx] *= scale

    return final_flows, node_to_idx, requests_list
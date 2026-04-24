import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import networkx as nx
from itertools import islice
from collections import defaultdict
import time
from typing import List, Dict, Tuple, Any, Optional, Set, Union

class PPOAgent(nn.Module):
    """
    Нейросетевой агент для алгоритма PPO (Proximal Policy Optimization).
    
    Состоит из Actor (предсказывает распределение потоков по путям) 
    и Critic (оценивает ценность текущего состояния сети).
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Args:
            state_dim: Размерность вектора состояния сети.
            action_dim: Количество доступных действий (путей для заявок).
            hidden_dim: Количество нейронов в скрытых слоях.
        """
        super(PPOAgent, self).__init__()
        # Сеть Актора: определяет среднее значение (mean) для распределения действий
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # Обучаемый параметр стандартного отклонения для исследования среды
        self.actor_log_std = nn.Parameter(torch.ones(1, action_dim) * -0.5)

        # Сеть Критика: предсказывает V(s) - ожидаемую суммарную награду
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Прямой проход сети."""
        mean = self.actor_mean(state)
        log_std = self.actor_log_std.expand_as(mean)
        std = torch.exp(log_std)
        value = self.critic(state)
        return mean, std, value


class PowerRoutingEnv:
    """
    Среда симуляции электросети для обучения RL-агента.
    
    Реализует интерфейс, похожий на OpenAI Gym: state, reward, done.
    Внутри инкапсулирована логика Water-filling для соблюдения лимитов.
    """
    def __init__(
        self, 
        nodes: List[str], 
        caps: Dict[Tuple[str, str], float], 
        req_list: List[Dict[str, Any]], 
        paths_per_req: List[List[List[str]]], 
        K_paths: int, 
        max_steps: int = 32
    ):
        """
        Args:
            nodes: Список узлов.
            caps: Лимиты ребер.
            req_list: Список обработанных заявок.
            paths_per_req: Список k-кратчайших путей для каждой заявки.
            K_paths: Фиксированное число путей на одну заявку.
            max_steps: Лимит шагов в одном эпизоде обучения.
        """
        self.nodes = nodes
        self.caps = caps
        self.req_list = req_list
        self.paths_per_req = paths_per_req
        self.K_paths = K_paths
        self.max_steps = max_steps

        self.edges = list(caps.keys())
        self.edge_idx = {e: i for i, e in enumerate(self.edges)}
        self.num_edges = len(self.edges)

        self.cap_arr = np.array([caps.get(e, 1e-5) for e in self.edges])
        self.max_cap = np.max(self.cap_arr) if self.num_edges > 0 else 1.0

        self.num_reqs = len(req_list)
        self.demands = np.array([r['amount'] for r in self.req_list])
        self.max_demand = np.max(self.demands) if self.num_reqs > 0 else 1.0
        self.total_demand = np.sum(self.demands) if self.num_reqs > 0 else 1.0

        self.step_count = 0
        self.logits = np.zeros((self.num_reqs, self.K_paths))
        self.current_metric = 0.0
        self.edge_loads = np.zeros(self.num_edges)
        self.actual_flows = np.zeros((self.num_reqs, self.K_paths))
        self.total_delivered = 0.0

        # Предварительный расчет маппинга ребер на пути для ускорения вычислений
        self._edge_req_paths = defaultdict(list)
        for r in range(self.num_reqs):
            for k in range(self.req_list[r]['actual_k']):
                path = self.paths_per_req[r][k]
                for i in range(len(path) - 1):
                    e = (path[i], path[i + 1])
                    if e in self.edge_idx:
                        self._edge_req_paths[self.edge_idx[e]].append((r, k))

    def reset(self) -> np.ndarray:
        """Сброс среды в начальное состояние."""
        self.step_count = 0
        self.logits = np.zeros((self.num_reqs, self.K_paths))
        self.current_metric, self.total_delivered, self.edge_loads, self.actual_flows = self._evaluate_flow(self.logits)
        return self._get_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Выполнение шага в среде.
        
        Args:
            action: Дельта изменений для распределения потоков (логитов).
        """
        action = action.reshape((self.num_reqs, self.K_paths))
        self.logits += action * 0.2
        self.logits = np.clip(self.logits, -10.0, 10.0)
        self.step_count += 1

        new_metric, new_delivered, new_edge_loads, new_actual_flows = self._evaluate_flow(self.logits)
        reward = new_metric - self.current_metric

        self.current_metric = new_metric
        self.total_delivered = new_delivered
        self.edge_loads = new_edge_loads
        self.actual_flows = new_actual_flows

        done = (self.step_count >= self.max_steps)
        return self._get_state(), reward, done, {}

    def _get_probs(self, logits: np.ndarray) -> np.ndarray:
        """Расчет вероятностей путей через Softmax с маскированием."""
        masked_logits = logits.copy()
        for r in range(self.num_reqs):
            actual_k = self.req_list[r]['actual_k']
            for k in range(actual_k, self.K_paths):
                masked_logits[r, k] = -1e9

        max_logits = np.max(masked_logits, axis=1, keepdims=True)
        exp_L = np.exp(masked_logits - max_logits)
        return exp_L / np.sum(exp_L, axis=1, keepdims=True)

    def _evaluate_flow(self, logits: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Ядро среды: оценивает физические потоки и соблюдение лимитов.
        Использует многопроходный Water-filling для обеспечения Fairness.
        """
        probs = self._get_probs(logits)
        actual_flows = np.zeros((self.num_reqs, self.K_paths))
        for r in range(self.num_reqs):
            actual_flows[r] = probs[r] * self.req_list[r]['amount']

        for _ in range(15):
            edge_loads = np.zeros(self.num_edges)
            for r in range(self.num_reqs):
                for k in range(self.req_list[r]['actual_k']):
                    flow = actual_flows[r, k]
                    if flow <= 0: continue
                    path = self.paths_per_req[r][k]
                    for i in range(len(path) - 1):
                        e = (path[i], path[i + 1])
                        if e in self.edge_idx:
                            edge_loads[self.edge_idx[e]] += flow

            overloaded = False
            scaling = np.ones((self.num_reqs, self.K_paths))

            for e_idx in range(self.num_edges):
                if edge_loads[e_idx] <= self.cap_arr[e_idx] + 1e-4:
                    continue

                overloaded = True
                cap = self.cap_arr[e_idx]
                req_flow_on_edge = defaultdict(float)
                req_paths_on_edge = defaultdict(list)

                for (r, k) in self._edge_req_paths[e_idx]:
                    flow = actual_flows[r, k]
                    if flow > 0:
                        req_flow_on_edge[r] += flow
                        req_paths_on_edge[r].append(k)

                if not req_flow_on_edge: continue

                remaining_cap = cap
                uncapped = set(req_flow_on_edge.keys())
                final_alloc = {}

                while uncapped and remaining_cap > 1e-6:
                    total_demand_uncapped = sum(self.req_list[r]['amount'] for r in uncapped)
                    if total_demand_uncapped <= 0: break
                    fitted = set()
                    for r in uncapped:
                        share = remaining_cap * (self.req_list[r]['amount'] / total_demand_uncapped)
                        if req_flow_on_edge[r] <= share + 1e-6:
                            final_alloc[r] = req_flow_on_edge[r]
                            fitted.add(r)
                    if not fitted:
                        for r in uncapped:
                            final_alloc[r] = remaining_cap * (self.req_list[r]['amount'] / total_demand_uncapped)
                        break
                    for r in fitted:
                        remaining_cap -= final_alloc[r]
                        uncapped.remove(r)

                for r, alloc in final_alloc.items():
                    actual = req_flow_on_edge[r]
                    if actual > alloc + 1e-6:
                        local_scale = alloc / actual
                        for k in req_paths_on_edge[r]:
                            scaling[r, k] = min(scaling[r, k], local_scale)

            actual_flows *= scaling
            if not overloaded: break

        total_delivered = np.sum(actual_flows)
        delivered_ratios = np.zeros(self.num_reqs)
        for r in range(self.num_reqs):
            deliv_r = np.sum(actual_flows[r, :self.req_list[r]['actual_k']])
            delivered_ratios[r] = deliv_r / (self.req_list[r]['amount'] + 1e-5)

        # Метрика награды: Общий объем / Потери - Штраф за дисбаланс (Variance)
        variance_penalty = np.var(delivered_ratios)
        normalized_delivered = total_delivered / self.total_demand
        metric = normalized_delivered - 0.5 * variance_penalty

        final_edge_loads = np.zeros(self.num_edges)
        for r in range(self.num_reqs):
            for k in range(self.req_list[r]['actual_k']):
                flow = actual_flows[r, k]
                path = self.paths_per_req[r][k]
                for i in range(len(path) - 1):
                    e = (path[i], path[i + 1])
                    if e in self.edge_idx:
                        final_edge_loads[self.edge_idx[e]] += flow

        return metric, total_delivered, final_edge_loads, actual_flows

    def _get_state(self) -> np.ndarray:
        """Формирует вектор состояния для нейросети."""
        norm_loads = self.edge_loads / (self.cap_arr + 1e-5)
        probs = self._get_probs(self.logits).flatten()
        norm_demands = self.demands / self.max_demand
        norm_caps = self.cap_arr / self.max_cap
        return np.concatenate([norm_loads, probs, norm_demands, norm_caps]).astype(np.float32)

    def get_final_flows(self) -> Dict[str, Any]:
        """Возвращает результаты в формате, совместимом с UI."""
        delivered_dict = {}
        load_distribution = defaultdict(float)
        request_flows = defaultdict(lambda: defaultdict(float))

        for r in range(self.num_reqs):
            src, dst = self.req_list[r]['src'], self.req_list[r]['dst']
            req_key = (src, dst)
            delivered_r = np.sum(self.actual_flows[r, :self.req_list[r]['actual_k']])
            delivered_dict[req_key] = delivered_dict.get(req_key, 0.0) + delivered_r
            
            for k in range(self.req_list[r]['actual_k']):
                flow = self.actual_flows[r, k]
                if flow <= 0: continue
                path = self.paths_per_req[r][k]
                for i in range(len(path) - 1):
                    e = (path[i], path[i + 1])
                    load_distribution[e] += flow
                    request_flows[req_key][e] += flow

        return {
            'load_distribution': dict(load_distribution),
            'delivered': delivered_dict,
            'request_flows': {req: dict(flows) for req, flows in request_flows.items()}
        }


def get_k_shortest_paths(G: nx.DiGraph, source: str, target: str, k: int = 3) -> List[List[str]]:
    """Находит k кратчайших простых путей между узлами."""
    try:
        return list(islice(nx.shortest_simple_paths(G, source, target), k))
    except nx.NetworkXNoPath:
        return []


def run_rl(
    nodes: List[str], 
    dests: List[str], 
    adj: Dict[str, List[str]], 
    caps: Dict[Tuple[str, str], float], 
    reqs: Dict[Tuple[str, str], float], 
    epochs: Optional[int] = None, 
    time_limit_min: int = 5, 
    early_stop: bool = True, 
    K_paths: int = 15, 
    gamma: float = 0.99,
    lr: float = 3e-4
) -> Dict[str, Any]:
    """
    Запускает процесс обучения RL-агента для управления потоками.
    
    Алгоритм динамически подбирает параметры обучения под сложность сети
    и оптимизирует распределение мощностей по набору k-кратчайших путей.
    """
    start_time = time.time()
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for u in nodes:
        for v in adj.get(u, []):
            G.add_edge(u, v, capacity=caps.get((u, v), 1.0))

    req_list, paths_per_req = [], []
    for (src, dst), amount in reqs.items():
        if src in G.nodes and dst in G.nodes:
            paths = get_k_shortest_paths(G, src, dst, k=K_paths)
            if paths:
                actual_k = len(paths)
                while len(paths) < K_paths: paths.append([])
                req_list.append({'src': src, 'dst': dst, 'amount': amount, 'actual_k': actual_k})
                paths_per_req.append(paths)

    num_requests = len(req_list)
    if num_requests == 0: return {'load_distribution': {}, 'delivered': {}}

    # Dynamic Scaling сложности обучения
    complexity = len(caps) * num_requests
    if complexity < 50:
        auto_epochs, max_steps, n_episodes = 60, 12, 10
    elif complexity < 500:
        auto_epochs, max_steps, n_episodes = 120, 16, 16
    else:
        auto_epochs, max_steps, n_episodes = 200, 20, 20

    epochs = epochs if epochs is not None else auto_epochs
    batch_size = max_steps * n_episodes

    env = PowerRoutingEnv(nodes, caps, req_list, paths_per_req, K_paths, max_steps=max_steps)
    eval_env = PowerRoutingEnv(nodes, caps, req_list, paths_per_req, K_paths, max_steps=max_steps)

    state_dim = env.num_edges + (env.num_reqs * K_paths) + env.num_reqs + env.num_edges
    action_dim = env.num_reqs * K_paths

    agent = PPOAgent(state_dim, action_dim)
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_metric_val, best_result = -float('inf'), None
    no_improvement_count = 0
    state = env.reset()

    for epoch in range(epochs):
        if time.time() - start_time > time_limit_min * 60: break

        # 1. Сбор Rollout
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        for _ in range(batch_size):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                mean, std, value = agent(state_tensor)
                dist = Normal(mean, std)
                action = dist.sample()
                log_p = dist.log_prob(action).sum(dim=-1)

            action_np = action.squeeze(0).numpy()
            next_state, reward, done, _ = env.step(action_np)

            states.append(state); actions.append(action_np); log_probs.append(log_p.item())
            rewards.append(reward); values.append(value.item()); dones.append(done)

            state = next_state if not done else env.reset()

        # 2. Evaluation
        eval_state = eval_env.reset()
        for _ in range(eval_env.max_steps):
            with torch.no_grad():
                mean, _, _ = agent(torch.FloatTensor(eval_state).unsqueeze(0))
            eval_state, _, d, _ = eval_env.step(mean.squeeze(0).numpy())
            if d: break

        if eval_env.current_metric > best_metric_val + 1e-6:
            best_metric_val, best_result = eval_env.current_metric, eval_env.get_final_flows()
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if early_stop and no_improvement_count >= 20: break

        # 3. PPO Update (GAE)
        returns, advantages, gae, lam = [], [], 0, 0.95
        with torch.no_grad():
            _, _, next_val = agent(torch.FloatTensor(state).unsqueeze(0))
            next_val = next_val.item()

        for i in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[i]
            next_value = next_val if i == len(rewards) - 1 else values[i + 1]
            delta = rewards[i] + gamma * next_value * next_non_terminal - values[i]
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages.insert(0, gae); returns.insert(0, gae + values[i])

        states_t, actions_t = torch.FloatTensor(np.array(states)), torch.FloatTensor(np.array(actions))
        old_log_probs_t, returns_t = torch.FloatTensor(np.array(log_probs)), torch.FloatTensor(np.array(returns))
        advantages_t = torch.FloatTensor(np.array(advantages))
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        for _ in range(4):
            mean, std, value = agent(states_t)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions_t).sum(dim=-1)
            ratio = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages_t
            loss = -torch.min(surr1, surr2).mean() + 0.5 * nn.MSELoss()(value.squeeze(-1), returns_t) - 0.01 * dist.entropy().mean()
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5); optimizer.step()

        scheduler.step()
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:4d} | Metric: {eval_env.current_metric:.4f} | Delivered: {eval_env.total_delivered:.1f} kW")

    return best_result if best_result else {'load_distribution': {}, 'delivered': {}}
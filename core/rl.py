import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import networkx as nx
from itertools import islice
from collections import defaultdict


class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PPOAgent, self).__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.actor_log_std = nn.Parameter(torch.ones(1, action_dim) * -0.5)
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        mean = self.actor_mean(state)
        log_std = self.actor_log_std.expand_as(mean)
        std = torch.exp(log_std)
        value = self.critic(state)
        return mean, std, value


class PowerRoutingEnv:
    def __init__(self, nodes, caps, req_list, paths_per_req, K_paths, max_steps=32):
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

    def reset(self):
        self.step_count = 0
        self.logits = np.zeros((self.num_reqs, self.K_paths))
        self.current_metric, self.total_delivered, self.edge_loads, self.actual_flows = self._evaluate_flow(self.logits)
        return self._get_state()

    def step(self, action):
        action = action.reshape((self.num_reqs, self.K_paths))
        self.logits += action
        # Предотвращаем взрыв логитов (Исчезающий градиент Softmax)
        self.logits = np.clip(self.logits, -10.0, 10.0)
        self.step_count += 1
        
        new_metric, new_delivered, new_edge_loads, new_actual_flows = self._evaluate_flow(self.logits)
        
        # Награда: приращение интегральной метрики (throughput - penalty)
        reward = new_metric - self.current_metric
        
        self.current_metric = new_metric
        self.total_delivered = new_delivered
        self.edge_loads = new_edge_loads
        self.actual_flows = new_actual_flows
        
        done = (self.step_count >= self.max_steps)
        return self._get_state(), reward, done, {}

    def _get_probs(self, logits):
        masked_logits = logits.copy()
        for r in range(self.num_reqs):
            actual_k = self.req_list[r]['actual_k']
            for k in range(actual_k, self.K_paths):
                masked_logits[r, k] = -1e9 
                
        max_logits = np.max(masked_logits, axis=1, keepdims=True)
        exp_L = np.exp(masked_logits - max_logits)
        return exp_L / np.sum(exp_L, axis=1, keepdims=True)

    def _evaluate_flow(self, logits):
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
                    for i in range(len(path)-1):
                        e = (path[i], path[i+1])
                        if e in self.edge_idx:
                            edge_loads[self.edge_idx[e]] += flow
                            
            overloaded = False
            path_ratios = np.ones((self.num_reqs, self.K_paths))
            
            for r in range(self.num_reqs):
                for k in range(self.req_list[r]['actual_k']):
                    path = self.paths_per_req[r][k]
                    min_ratio = 1.0
                    for i in range(len(path)-1):
                        e = (path[i], path[i+1])
                        if e in self.edge_idx:
                            e_idx = self.edge_idx[e]
                            if edge_loads[e_idx] > self.cap_arr[e_idx] + 1e-4:
                                overloaded = True
                                ratio = self.cap_arr[e_idx] / edge_loads[e_idx]
                                if ratio < min_ratio:
                                    min_ratio = ratio
                    path_ratios[r, k] = min_ratio
            
            actual_flows *= path_ratios
            
            if not overloaded:
                break
                
        total_delivered = np.sum(actual_flows)
        
        delivered_ratios = np.zeros(self.num_reqs)
        for r in range(self.num_reqs):
            deliv_r = np.sum(actual_flows[r, :self.req_list[r]['actual_k']])
            delivered_ratios[r] = deliv_r / (self.req_list[r]['amount'] + 1e-5)
            
        variance_penalty = np.var(delivered_ratios)
        normalized_delivered = total_delivered / self.total_demand
        
        metric = normalized_delivered - 0.5 * variance_penalty
        
        final_edge_loads = np.zeros(self.num_edges)
        for r in range(self.num_reqs):
            for k in range(self.req_list[r]['actual_k']):
                flow = actual_flows[r, k]
                path = self.paths_per_req[r][k]
                for i in range(len(path)-1):
                    e = (path[i], path[i+1])
                    if e in self.edge_idx:
                        final_edge_loads[self.edge_idx[e]] += flow
                        
        return metric, total_delivered, final_edge_loads, actual_flows

    def _get_state(self):
        norm_loads = self.edge_loads / (self.cap_arr + 1e-5)
        probs = self._get_probs(self.logits).flatten()
        norm_demands = self.demands / self.max_demand
        norm_caps = self.cap_arr / self.max_cap
        return np.concatenate([norm_loads, probs, norm_demands, norm_caps]).astype(np.float32)

    def get_final_flows(self):
        delivered_dict = {}
        load_distribution = defaultdict(float)
        for r in range(self.num_reqs):
            src = self.req_list[r]['src']
            dst = self.req_list[r]['dst']
            delivered_r = 0.0
            for k in range(self.req_list[r]['actual_k']):
                flow = self.actual_flows[r, k]
                if flow <= 0: continue
                delivered_r += flow
                path = self.paths_per_req[r][k]
                for i in range(len(path)-1):
                    e = (path[i], path[i+1])
                    load_distribution[e] += flow
            delivered_dict[(src, dst)] = delivered_dict.get((src, dst), 0.0) + delivered_r
        return {'load_distribution': dict(load_distribution), 'delivered': delivered_dict}


def get_k_shortest_paths(G, source, target, k=3):
    try:
        paths = list(islice(nx.shortest_simple_paths(G, source, target), k))
        return paths
    except nx.NetworkXNoPath:
        return []


def run_rl(nodes, dests, adj, caps, reqs, epochs=None, K_paths=5, gamma=0.99, lr=3e-4):
    G = nx.DiGraph()
    for u in nodes:
        for v in adj.get(u, []):
            cap = caps.get((u, v), 1.0)
            G.add_edge(u, v, capacity=cap)

    req_list = []
    paths_per_req = []

    for (src, dst), amount in reqs.items():
        if src in nodes and dst in nodes:
            paths = get_k_shortest_paths(G, src, dst, k=K_paths)
            if paths:
                actual_k = len(paths)
                while len(paths) < K_paths:
                    paths.append([])
                req_list.append({'src': src, 'dst': dst, 'amount': amount, 'actual_k': actual_k})
                paths_per_req.append(paths)

    num_requests = len(req_list)
    if num_requests == 0:
        return {'load_distribution': {}, 'delivered': {}}

    # Автоматический подбор гиперпараметров (Dynamic Scaling)
    complexity = len(caps) * num_requests
    if complexity < 50:      # Очень простая сеть (например, базовая таблица 1.2)
        auto_epochs = 50
        batch_size = 32
        max_steps = 8
    elif complexity < 500:   # Средняя сеть
        auto_epochs = 100
        batch_size = 128
        max_steps = 16
    else:                    # Сложная сеть (eval_complex.py)
        auto_epochs = 200
        batch_size = 256
        max_steps = 32
        
    # Если пользователь явно передал epochs, используем его, иначе авто
    epochs = epochs if epochs is not None else auto_epochs

    env = PowerRoutingEnv(nodes, caps, req_list, paths_per_req, K_paths, max_steps=max_steps)
    # Отдельная среда для детерминированной оценки (Evaluation Mode)
    eval_env = PowerRoutingEnv(nodes, caps, req_list, paths_per_req, K_paths, max_steps=max_steps)
    
    state_dim = env.num_edges + (env.num_reqs * K_paths) + env.num_reqs + env.num_edges
    action_dim = env.num_reqs * K_paths
    
    agent = PPOAgent(state_dim, action_dim, hidden_dim=128)
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    
    # epochs теперь означает количество обновлений PPO (PPO Update Epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_metric_val = -float('inf')
    best_result = None
    best_delivered_kwt = 0.0
    
    state = env.reset()

    for epoch in range(epochs):
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        
        # 1. Сбор данных (Rollout)
        for _ in range(batch_size):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                mean, std, value = agent(state_tensor)
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
            
            action_np = action.squeeze(0).numpy()
            next_state, reward, done, _ = env.step(action_np)
            
            states.append(state)
            actions.append(action_np)
            log_probs.append(log_prob.item())
            rewards.append(reward)
            values.append(value.item())
            dones.append(done)
            
            state = next_state
            
            if done:
                state = env.reset()
                
        # 2. Оценка детерминированной политики (Evaluation Mode без шума)
        eval_state = eval_env.reset()
        for _ in range(eval_env.max_steps):
            state_tensor = torch.FloatTensor(eval_state).unsqueeze(0)
            with torch.no_grad():
                mean, _, _ = agent(state_tensor)
            eval_state, _, eval_done, _ = eval_env.step(mean.squeeze(0).numpy())
            if eval_done:
                break
                
        if eval_env.current_metric > best_metric_val:
            best_metric_val = eval_env.current_metric
            best_result = eval_env.get_final_flows()
            best_delivered_kwt = eval_env.total_delivered
                
        # 3. Обновление PPO (GAE)
        returns = []
        advantages = []
        gae = 0
        lam = 0.95
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            _, _, next_val = agent(state_tensor)
            next_val = next_val.item()
            
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[i]
                next_value = next_val
            else:
                next_non_terminal = 1.0 - dones[i]
                next_value = values[i+1]
                
            delta = rewards[i] + gamma * next_value * next_non_terminal - values[i]
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
            
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.FloatTensor(np.array(actions))
        old_log_probs_t = torch.FloatTensor(np.array(log_probs))
        returns_t = torch.FloatTensor(np.array(returns))
        advantages_t = torch.FloatTensor(np.array(advantages))
        
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # Обновление сети 4 раза на собранном батче
        for _ in range(4):
            mean, std, value = agent(states_t)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions_t).sum(dim=-1)
            ratio = torch.exp(new_log_probs - old_log_probs_t)
            
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages_t
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = nn.MSELoss()(value.squeeze(-1), returns_t)
            entropy = dist.entropy().mean()
            
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
            
        scheduler.step()
        
        # Логирование прогресса каждые 10% от общего числа эпох (или на первой эпохе)
        print_interval = max(1, epochs // 10)
        if (epoch + 1) % print_interval == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{epochs} | Eval Metric (Fairness): {eval_env.current_metric:.4f} | Eval Delivered: {eval_env.total_delivered:.1f} kW | Best Delivered: {best_delivered_kwt:.1f} kW")

    return best_result if best_result else {'load_distribution': {}, 'delivered': {}}
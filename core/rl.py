import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import networkx as nx
from itertools import islice
from collections import defaultdict


class PPOAgent(nn.Module):
    """
    Агент PPO (Proximal Policy Optimization) для задачи маршрутизации.
    Оценивает состояние сети и выдает изменения (deltas) для логитов путей.
    """
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
        # Обучаемое стандартное отклонение (начинаем с меньшей дисперсии для стабильности)
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
    """
    Среда обучения (MDP) для последовательной корректировки потоков.
    Агент делает T шагов, корректируя распределение потоков по путям.
    """
    def __init__(self, nodes, caps, req_list, paths_per_req, K_paths, max_steps=10):
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
        self.num_reqs = len(req_list)
        
        self.step_count = 0
        self.logits = np.zeros((self.num_reqs, self.K_paths))
        self.current_delivered = 0.0
        self.edge_loads = np.zeros(self.num_edges)

    def reset(self):
        self.step_count = 0
        # Инициализация равными вероятностями
        self.logits = np.zeros((self.num_reqs, self.K_paths))
        self.current_delivered, self.edge_loads = self._evaluate_flow(self.logits)
        return self._get_state()

    def step(self, action):
        """ Применение действия агента (изменение логитов) """
        # action shape is (num_reqs * K_paths)
        action = action.reshape((self.num_reqs, self.K_paths))
        self.logits += action
        self.step_count += 1
        
        new_delivered, new_edge_loads = self._evaluate_flow(self.logits)
        
        # Награда: приращение доставленной энергии
        reward = new_delivered - self.current_delivered
        
        self.current_delivered = new_delivered
        self.edge_loads = new_edge_loads
        
        done = (self.step_count >= self.max_steps)
        return self._get_state(), reward, done, {}

    def _get_probs(self, logits):
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_L = np.exp(logits - max_logits)
        return exp_L / np.sum(exp_L, axis=1, keepdims=True)

    def _evaluate_flow(self, logits):
        """ 
        Симуляция среды: вычисление реальной доставки энергии с учетом 
        пропорциональных ограничений сети при перегрузке.
        """
        probs = self._get_probs(logits)
        
        # 1. Расчет запрашиваемых (attempted) нагрузок на линии
        edge_loads = np.zeros(self.num_edges)
        for r in range(self.num_reqs):
            req_amount = self.req_list[r]['amount']
            for k in range(len(self.paths_per_req[r])):
                flow = probs[r, k] * req_amount
                path = self.paths_per_req[r][k]
                for i in range(len(path)-1):
                    e = (path[i], path[i+1])
                    if e in self.edge_idx:
                        edge_loads[self.edge_idx[e]] += flow
                        
        # 2. Применение жесткого пропорционального отсечения
        total_delivered = 0.0
        for r in range(self.num_reqs):
            req_amount = self.req_list[r]['amount']
            for k in range(len(self.paths_per_req[r])):
                flow = probs[r, k] * req_amount
                if flow <= 0: continue
                
                allowed_ratio = 1.0
                path = self.paths_per_req[r][k]
                # Поиск "бутылочного горлышка" (наименьшего отношения cap / load) на маршруте
                for i in range(len(path)-1):
                    e = (path[i], path[i+1])
                    if e in self.edge_idx:
                        e_idx = self.edge_idx[e]
                        if edge_loads[e_idx] > self.cap_arr[e_idx]:
                            ratio = self.cap_arr[e_idx] / edge_loads[e_idx]
                            if ratio < allowed_ratio:
                                allowed_ratio = ratio
                
                total_delivered += flow * allowed_ratio
                
        return total_delivered, edge_loads

    def _get_state(self):
        """ Формирование вектора состояния для нейросети """
        # Нормализованная загрузка линий
        norm_loads = self.edge_loads / (self.cap_arr + 1e-5)
        # Текущие вероятности выбора путей
        probs = self._get_probs(self.logits).flatten()
        return np.concatenate([norm_loads, probs]).astype(np.float32)

    def get_final_flows(self):
        """ Возвращает финальное распределение для интерфейса пользователя """
        probs = self._get_probs(self.logits)
        
        edge_loads = np.zeros(self.num_edges)
        for r in range(self.num_reqs):
            req_amount = self.req_list[r]['amount']
            for k in range(len(self.paths_per_req[r])):
                flow = probs[r, k] * req_amount
                path = self.paths_per_req[r][k]
                for i in range(len(path)-1):
                    e = (path[i], path[i+1])
                    if e in self.edge_idx:
                        edge_loads[self.edge_idx[e]] += flow
                        
        delivered_dict = {}
        load_distribution = defaultdict(float)
        
        for r in range(self.num_reqs):
            src = self.req_list[r]['src']
            dst = self.req_list[r]['dst']
            req_amount = self.req_list[r]['amount']
            delivered_r = 0.0
            
            for k in range(len(self.paths_per_req[r])):
                flow = probs[r, k] * req_amount
                if flow <= 0: continue
                
                allowed_ratio = 1.0
                path = self.paths_per_req[r][k]
                for i in range(len(path)-1):
                    e = (path[i], path[i+1])
                    if e in self.edge_idx:
                        e_idx = self.edge_idx[e]
                        if edge_loads[e_idx] > self.cap_arr[e_idx]:
                            ratio = self.cap_arr[e_idx] / edge_loads[e_idx]
                            if ratio < allowed_ratio:
                                allowed_ratio = ratio
                
                deliv = flow * allowed_ratio
                delivered_r += deliv
                
                for i in range(len(path)-1):
                    e = (path[i], path[i+1])
                    load_distribution[e] += deliv
                    
            delivered_dict[(src, dst)] = delivered_dict.get((src, dst), 0.0) + delivered_r
            
        return {'load_distribution': dict(load_distribution), 'delivered': delivered_dict}


def get_k_shortest_paths(G, source, target, k=3):
    try:
        paths = list(islice(nx.shortest_simple_paths(G, source, target), k))
        return paths
    except nx.NetworkXNoPath:
        return []


def run_rl(nodes, dests, adj, caps, reqs, epochs=1000, K_paths=5, gamma=0.99, lr=3e-4):
    """
    Главная функция запуска Deep RL (PPO).
    Создает среду и обучает агента управлять потоками.
    """
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
                # Паддинг путей, если их меньше K
                while len(paths) < K_paths:
                    paths.append(paths[0])
                req_list.append({'src': src, 'dst': dst, 'amount': amount})
                paths_per_req.append(paths)

    num_requests = len(req_list)
    if num_requests == 0:
        return {'load_distribution': {}, 'delivered': {}}

    env = PowerRoutingEnv(nodes, caps, req_list, paths_per_req, K_paths, max_steps=8)
    
    state_dim = env.num_edges + env.num_reqs * K_paths
    action_dim = env.num_reqs * K_paths
    
    agent = PPOAgent(state_dim, action_dim)
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_total_delivered = -1
    best_result = None

    # Цикл обучения PPO
    for epoch in range(epochs):
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        state = env.reset()
        
        # Сбор траектории
        for t in range(env.max_steps):
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
            
            # Сохранение лучшего найденного решения
            if env.current_delivered > best_total_delivered:
                best_total_delivered = env.current_delivered
                best_result = env.get_final_flows()
                
        # GAE (Generalized Advantage Estimation)
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
        
        # Нормализация advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # Обновление PPO (K эпох)
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
            
            # Общий loss (поощряем энтропию для исследования)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
            
        scheduler.step()

    return best_result if best_result else {'load_distribution': {}, 'delivered': {}}
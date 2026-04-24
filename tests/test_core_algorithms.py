import pytest
import numpy as np
import time
from typing import List, Dict, Tuple, Any, Callable
from core.aco import run_aco
from core.rl import run_rl
from core.gnn import run_gnn
from utils.generate_dataset import generate_synthetic_network
from core.data_loader import load_network_data

# Список алгоритмов для параметризации
ALGORITHMS = [
    ("ACO", lambda n, d, a, c, r: run_aco(n, d, a, c, r, n_iterations=10, early_stop=False)),
    ("RL", lambda n, d, a, c, r: run_rl(n, d, a, c, r, epochs=10, early_stop=False)),
    ("GNN", lambda n, d, a, c, r: run_gnn(n, c, r, epochs=10, early_stop=False))
]


@pytest.mark.parametrize("name, run_func", ALGORITHMS)
def test_algorithm_fairness_and_proportionality(name: str, run_func: Callable) -> None:
    """
    Проверка справедливости (Water-filling/Proportionality).
    Два поставщика (100 и 200 кВт) претендуют на одну линию с лимитом 150 кВт.
    Ожидается, что их обрежут пропорционально (50 и 100 кВт).
    """
    nodes = ['S1', 'S2', 'Hub', 'Consumer']
    dests = ['Consumer']
    adj = {
        'S1': ['Hub'],
        'S2': ['Hub'],
        'Hub': ['Consumer'],
        'Consumer': []
    }
    caps = {
        ('S1', 'Hub'): 1000.0,
        ('S2', 'Hub'): 1000.0,
        ('Hub', 'Consumer'): 150.0  # Узкое горлышко
    }
    requests = {
        ('S1', 'Consumer'): 100.0,
        ('S2', 'Consumer'): 200.0
    }

    res = run_func(nodes, dests, adj, caps, requests)
    
    delivered = res['delivered']
    s1_del = delivered.get(('S1', 'Consumer'), 0.0)
    s2_del = delivered.get(('S2', 'Consumer'), 0.0)

    # Проверка лимита: суммарно не более 150
    assert s1_del + s2_del <= 150.0 + 1e-3, f"{name}: Превышен лимит узкого горлышка"
    
    # Проверка пропорциональности (1 к 2)
    if s1_del > 0 and s2_del > 0:
        ratio = s2_del / s1_del
        assert 1.8 <= ratio <= 2.2, f"{name}: Нарушена пропорциональность (ratio={ratio:.2f}, ожидалось ~2.0)"


@pytest.mark.parametrize("name, run_func", ALGORITHMS)
def test_algorithm_redirection_to_alt_path(name: str, run_func: Callable) -> None:
    """
    Проверка перенаправления (Redirection).
    Прямой путь ограничен, но есть обходной путь с большой мощностью.
    """
    nodes = ['Src', 'A', 'B', 'Dst']
    dests = ['Dst']
    adj = {
        'Src': ['A', 'B'],
        'A': ['Dst'],
        'B': ['Dst'],
        'Dst': []
    }
    caps = {
        ('Src', 'A'): 10.0,
        ('A', 'Dst'): 10.0,
        ('Src', 'B'): 100.0,
        ('B', 'Dst'): 100.0
    }
    requests = {
        ('Src', 'Dst'): 80.0
    }

    res = run_func(nodes, dests, adj, caps, requests)
    delivered = res['delivered'].get(('Src', 'Dst'), 0.0)

    assert delivered > 50.0, f"{name}: Алгоритм не нашел обходной путь"


@pytest.mark.parametrize("name, run_func", ALGORITHMS)
def test_algorithm_conservation_of_energy(name: str, run_func: Callable) -> None:
    """
    Проверка сохранения энергии.
    Сумма входящих потоков в узел должна быть равна сумме исходящих.
    """
    nodes = ['S', 'H1', 'H2', 'D']
    dests = ['D']
    adj = {'S': ['H1', 'H2'], 'H1': ['D'], 'H2': ['D'], 'D': []}
    caps = {e: 100.0 for e in [('S', 'H1'), ('S', 'H2'), ('H1', 'D'), ('H2', 'D')]}
    requests = {('S', 'D'): 50.0}

    res = run_func(nodes, dests, adj, caps, requests)
    load = res['load_distribution']
    
    flow_out_s = load.get(('S', 'H1'), 0.0) + load.get(('S', 'H2'), 0.0)
    flow_in_d = load.get(('H1', 'D'), 0.0) + load.get(('H2', 'D'), 0.0)
    
    assert abs(flow_out_s - flow_in_d) < 1e-3, f"{name}: Нарушен баланс энергии"
    assert abs(flow_out_s - res['delivered'][('S', 'D')]) < 1e-3, f"{name}: Несовпадение доставленного и потоков"


@pytest.mark.parametrize("name, run_func", ALGORITHMS)
def test_algorithm_high_load_stress(name: str, run_func: Callable) -> None:
    """
    Стресс-тест: Высокая нагрузка на реальной топологии Альфа.
    Проверка скорости и устойчивости при множественных узких местах.
    """
    df_req, df_cap = generate_synthetic_network(num_reqs=50, load_level=200.0, bottleneck_level=0.8)
    nodes, dests, adj, caps, requests, _ = load_network_data(df_req, df_cap)
    
    start_time = time.time()
    res = run_func(nodes, dests, adj, caps, requests)
    duration = time.time() - start_time
    
    assert res is not None
    assert 'delivered' in res
    assert duration < 30.0, f"{name}: Слишком медленная работа ({duration:.2f} сек)"
    
    for edge, load in res['load_distribution'].items():
        limit = caps.get(edge, float('inf'))
        assert load <= limit + 1e-2, f"{name}: Перегрузка в ребре {edge}"

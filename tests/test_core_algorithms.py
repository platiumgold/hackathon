import pytest
import numpy as np
from core.aco import run_aco
from core.rl import run_rl
from core.gnn import run_gnn

# Список алгоритмов для параметризации (чтобы один тест проверял всех сразу)
ALGORITHMS = [
    ("ACO", lambda n, d, a, c, r: run_aco(n, d, a, c, r, n_iterations=10, early_stop=False)),
    ("RL", lambda n, d, a, c, r: run_rl(n, d, a, c, r, epochs=10, early_stop=False)),
    ("GNN", lambda n, d, a, c, r: run_gnn(n, c, r, epochs=10, early_stop=False))
]

@pytest.mark.parametrize("name, run_func", ALGORITHMS)
def test_algorithm_fairness_and_proportionality(name, run_func):
    """
    ТЕСТ НА СПРАВЕДЛИВОСТЬ (Water-filling/Proportionality).
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

    # 1. Проверка лимита: суммарно не более 150
    assert s1_del + s2_del <= 150.0 + 1e-3, f"{name}: Превышен лимит узкого горлышка"
    
    # 2. Проверка пропорциональности (1 к 2)
    # Допускаем погрешность для ИИ-алгоритмов (5-10%), но они должны стремиться к 50/100
    if s1_del > 0 and s2_del > 0:
        ratio = s2_del / s1_del
        assert 1.8 <= ratio <= 2.2, f"{name}: Нарушена пропорциональность (ratio={ratio:.2f}, ожидалось ~2.0)"

@pytest.mark.parametrize("name, run_func", ALGORITHMS)
def test_algorithm_redirection_to_alt_path(name, run_func):
    """
    ТЕСТ НА ПЕРЕНАПРАВЛЕНИЕ (Redirection).
    Прямой путь ограничен, но есть длинный обходной путь с большой мощностью.
    Алгоритм должен использовать обходной путь, а не просто обрезать заявку.
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
        ('Src', 'A'): 10.0,   # Прямой путь почти закрыт
        ('A', 'Dst'): 10.0,
        ('Src', 'B'): 100.0,  # Обходной путь свободен
        ('B', 'Dst'): 100.0
    }
    requests = {
        ('Src', 'Dst'): 80.0
    }

    res = run_func(nodes, dests, adj, caps, requests)
    delivered = res['delivered'].get(('Src', 'Dst'), 0.0)

    # Должно быть доставлено больше 10 (значит обходной путь найден)
    assert delivered > 50.0, f"{name}: Алгоритм не нашел обходной путь (доставлено только {delivered})"

@pytest.mark.parametrize("name, run_func", ALGORITHMS)
def test_algorithm_conservation_of_energy(name, run_func):
    """
    ТЕСТ НА СОХРАНЕНИЕ ЭНЕРГИИ.
    Сумма входящих потоков в узел должна быть равна сумме исходящих.
    """
    nodes = ['S', 'H1', 'H2', 'D']
    dests = ['D']
    adj = {'S': ['H1', 'H2'], 'H1': ['D'], 'H2': ['D'], 'D': []}
    caps = {e: 100.0 for e in [('S', 'H1'), ('S', 'H2'), ('H1', 'D'), ('H2', 'D')]}
    requests = {('S', 'D'): 50.0}

    res = run_func(nodes, dests, adj, caps, requests)
    load = res['load_distribution']
    
    # Потоки из источника S
    flow_out_s = load.get(('S', 'H1'), 0.0) + load.get(('S', 'H2'), 0.0)
    # Потоки в сток D
    flow_in_d = load.get(('H1', 'D'), 0.0) + load.get(('H2', 'D'), 0.0)
    
    assert abs(flow_out_s - flow_in_d) < 1e-3, f"{name}: Нарушен баланс энергии"
    assert abs(flow_out_s - res['delivered'][('S', 'D')]) < 1e-3, f"{name}: Несовпадение доставленного и потоков"

import time
from utils.generate_dataset import generate_synthetic_network
from core.data_loader import load_network_data

@pytest.mark.parametrize("name, run_func", ALGORITHMS)
def test_algorithm_high_load_stress(name, run_func):
    """
    СТРЕСС-ТЕСТ: Высокая нагрузка на реальной топологии Альфа.
    Проверяем скорость работы и устойчивость при 50+ заявках и множественных узких местах.
    """
    # Генерируем 50 случайных заявок на реальной топологии
    df_req, df_cap = generate_synthetic_network(num_reqs=50, load_level=200.0, bottleneck_level=0.8)
    
    nodes, dests, adj, caps, requests, _ = load_network_data(df_req, df_cap)
    
    start_time = time.time()
    res = run_func(nodes, dests, adj, caps, requests)
    duration = time.time() - start_time
    
    # 1. Алгоритм не должен падать
    assert res is not None
    assert 'delivered' in res
    
    # 2. Время выполнения должно быть разумным (для хакатона < 30 сек)
    # GNN и ACO обычно быстрые, RL может быть дольше, но в тестах мы ограничили эпохи
    assert duration < 30.0, f"{name}: Слишком медленная работа ({duration:.2f} сек)"
    
    # 3. Физическая валидность на реальной сети
    for edge, load in res['load_distribution'].items():
        limit = caps.get(edge, float('inf'))
        assert load <= limit + 1e-2, f"{name}: Перегрузка на реальной топологии в ребре {edge}"

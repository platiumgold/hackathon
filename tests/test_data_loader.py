import pytest
import pandas as pd
from core.data_loader import get_logical_edges, get_current_topology

def test_logical_edges_detection():
    """Проверка корректности обнаружения 'логических' связей через скрытые узлы"""
    edges = get_logical_edges()
    # Пример: XVIII. -> g -> 5. XVIII. и 5 - видимые, g - скрытый. Должно быть XVIII. -> 5
    assert any(u == 'XVIII.' and v == '5' for u, v in edges)
    # Прямая связь тоже логическая
    assert any(u == 'A' and v == 'XVIII.' for u, v in edges)

def test_topology_reverse_logic():
    """Проверка корректности разворота графа при реверсе на пульте управления"""
    # Развернем главную магистраль A -> XVIII.
    reversed_edges = [('A', 'XVIII.')]
    new_topo = get_current_topology(reversed_edges)
    
    # Старая связь должна исчезнуть, новая (обратная) появиться
    assert ('XVIII.', 'A') in new_topo
    assert ('A', 'XVIII.') not in new_topo

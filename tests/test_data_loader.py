import pytest
import pandas as pd
from typing import List, Tuple
from core.data_loader import get_logical_edges, get_current_topology


def test_logical_edges_detection() -> None:
    """
    Проверка корректности обнаружения 'логических' связей через скрытые узлы.
    Проверяет, что алгоритм видит прямые связи и связи через технические узлы.
    """
    edges: List[Tuple[str, str]] = get_logical_edges()
    # Пример: XVIII. -> g -> 5. XVIII. и 5 - видимые, g - скрытый. Должно быть XVIII. -> 5
    assert any(u == 'XVIII.' and v == '5' for u, v in edges)
    # Прямая связь тоже логическая
    assert any(u == 'A' and v == 'XVIII.' for u, v in edges)


def test_topology_reverse_logic() -> None:
    """
    Проверка корректности разворота графа при реверсе на пульте управления.
    Проверяет, что старое ребро удаляется, а инвертированное добавляется в топологию.
    """
    # Развернем главную магистраль A -> XVIII.
    reversed_edges: List[Tuple[str, str]] = [('A', 'XVIII.')]
    new_topo: List[Tuple[str, str]] = get_current_topology(reversed_edges)
    
    # Старая связь должна исчезнуть, новая (обратная) появиться
    assert ('XVIII.', 'A') in new_topo
    assert ('A', 'XVIII.') not in new_topo

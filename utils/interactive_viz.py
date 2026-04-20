import plotly.graph_objects as go
import networkx as nx
from core.data_loader import TOPOLOGY_POS

def draw_interactive_heatmap(nodes, adj, capacities, final_load):
    fig = go.Figure()

    # Вспомогательная функция для цвета
    def get_color(ratio):
        r = int(255 * ratio)
        b = int(255 * (1 - ratio))
        return f'rgb({r}, 0, {b})'

    # Списки для средних точек (чтобы на них вешать текст и hover)
    mid_x = []
    mid_y = []
    mid_text = []

    # 1. Отрисовка ребер
    for (u, v), cap in capacities.items():
        if u not in TOPOLOGY_POS or v not in TOPOLOGY_POS: continue
        x0, y0 = TOPOLOGY_POS[u]
        x1, y1 = TOPOLOGY_POS[v]
        
        load = final_load.get((u, v), 0.0)
        ratio = min(load / cap, 1.0) if cap > 0 else 0.0
        
        color = 'lightgrey' if load < 0.1 else get_color(ratio)
        width = 1 if load < 0.1 else 4
        
        # Добавляем саму линию
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            line=dict(width=width, color=color),
            hoverinfo='none', # На линиях отключаем, чтобы не мешало
            mode='lines',
            showlegend=False
        ))

        # Вычисляем середину для текста
        if load >= 0.1: # Показываем текст только для активных ребер, чтобы не загромождать
            mid_x.append((x0 + x1) / 2)
            mid_y.append((y0 + y1) / 2)
            mid_text.append(f"{load:.1f}/{cap:.1f}")

    # 2. Отрисовка ТЕКСТА на ребрах (постоянно видимый)
    fig.add_trace(go.Scatter(
        x=mid_x, y=mid_y,
        mode='markers+text',
        text=mid_text,
        textposition="middle center",
        textfont=dict(size=9, color="black"),
        marker=dict(size=0, opacity=0), # Невидимые маркеры, только текст
        hoverinfo='text',
        hovertext=[f"Линия: {t}" for t in mid_text],
        showlegend=False
    ))

    # 3. Отрисовка узлов
    node_x, node_y, node_text = [], [], []
    node_marker_color, node_marker_size, node_marker_symbol = [], [], []

    for node in nodes:
        if node not in TOPOLOGY_POS: continue
        x, y = TOPOLOGY_POS[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"Узел: {node}")
        
        if str(node).isalpha(): # Источник
            node_marker_color.append('purple')
            node_marker_size.append(18)
            node_marker_symbol.append('square')
        elif str(node).isdigit(): # Потребитель
            node_marker_color.append('dodgerblue')
            node_marker_size.append(12)
            node_marker_symbol.append('circle')
        else: # Соединение
            node_marker_color.append('tomato')
            node_marker_size.append(22)
            node_marker_symbol.append('circle')

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[str(n) for n in nodes if n in TOPOLOGY_POS],
        textposition="top center",
        textfont=dict(size=10, color="black", family="Arial Black"),
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            color=node_marker_color,
            size=node_marker_size,
            symbol=node_marker_symbol,
            line=dict(color='black', width=1.5)),
        showlegend=False
    ))

    fig.update_layout(
        title='⚡ Интерактивная карта распределительной сети (Alpha)',
        title_font_size=24,
        dragmode='pan',
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template='plotly_white',
        height=800
    )

    return fig

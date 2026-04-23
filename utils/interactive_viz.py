import plotly.graph_objects as go
import math
from core.data_loader import TOPOLOGY_POS


def draw_interactive_heatmap(nodes, adj, capacities, total_load, selected_load=None):
    fig = go.Figure()

    def get_color(ratio):
        r = int(min(255, 255 * ratio))
        b = int(max(0, 255 * (1 - ratio)))
        return f'rgb({r}, 0, {b})'

    mid_x, mid_y, mid_text = [], [], []

    # 1. Отрисовка ребер и стрелок
    for (u, v), cap in capacities.items():
        if u not in TOPOLOGY_POS or v not in TOPOLOGY_POS: continue
        x0, y0 = TOPOLOGY_POS[u]
        x1, y1 = TOPOLOGY_POS[v]

        # Определяем направление перетока
        flow_uv = total_load.get((u, v), 0.0)
        flow_vu = total_load.get((v, u), 0.0)

        t_load = flow_uv + flow_vu
        s_load = (selected_load.get((u, v), 0.0) + selected_load.get((v, u),
                                                                     0.0)) if selected_load is not None else t_load

        ratio = min(t_load / cap, 1.2) if cap > 0 else 0.0
        color = 'lightgrey' if t_load < 0.1 else get_color(ratio)

        # Выравниваем размеры стрелок
        if selected_load is not None:
            width = 4 if s_load > 0.1 else 1.5
            opacity = 1.0 if s_load > 0.1 else 0.3
        else:
            width = 4 if t_load > 0.1 else 1.5
            opacity = 1.0

        # Линия
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            line=dict(width=width, color=color),
            opacity=opacity,
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))

        # HOVER-информация
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        mid_x.append(mx)
        mid_y.append(my)

        if selected_load is not None:
            mid_text.append(
                f"Участок: {u} ↔ {v}<br>Заявка: <b>{s_load:.1f} кВт</b><br>Всего: {t_load:.1f} кВт<br>Макс: {cap:.0f} кВт")
        else:
            mid_text.append(
                f"Участок: {u} ↔ {v}<br>Нагрузка: <b>{t_load:.1f} из {cap:.0f} кВт</b><br>Загруженность: {ratio * 100:.1f}%")

        # Вектор направления (по умолчанию по топологии схемы u -> v)
        start_x, start_y, end_x, end_y = x0, y0, x1, y1
        if flow_vu > flow_uv:
            start_x, start_y, end_x, end_y = x1, y1, x0, y0

        dx = end_x - start_x
        dy = end_y - start_y
        dist = math.hypot(dx, dy)

        # РИСУЕМ СТРЕЛКУ (более компактную и аккуратную)
        if dist > 0:
            nx, ny = dx / dist, dy / dist
            # Короткая стрелка, чтобы не перекрывать узлы
            arrow_len = min(0.2, dist * 0.1) 

            fig.add_annotation(
                x=mx + nx * arrow_len, y=my + ny * arrow_len,
                ax=mx - nx * arrow_len, ay=my - ny * arrow_len,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True,
                arrowhead=3, # Более изящный наконечник
                arrowsize=1, # В два раза меньше
                arrowwidth=min(2, width), # Тоньше
                arrowcolor=color,
                opacity=opacity * 0.8 # Чуть прозрачнее
            )

    # 2. Отрисовка невидимых точек для HOVER-текста
    if mid_x:
        fig.add_trace(go.Scatter(
            x=mid_x, y=mid_y,
            mode='markers',
            marker=dict(size=15, color='rgba(0,0,0,0)'),
            hoverinfo='text',
            hovertext=mid_text,
            hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial"),
            showlegend=False
        ))

    # 3. Отрисовка узлов
    node_x, node_y, node_text, node_labels = [], [], [], []
    node_marker_color, node_marker_size, node_marker_symbol = [], [], []

    for node in nodes:
        if node not in TOPOLOGY_POS: continue
        
        node_str = str(node)
        
        # СКРЫВАЕМ ТРАНЗИТНЫЕ УЗЛЫ (строчные латинские буквы)
        if node_str.islower() and node_str.isalpha():
            continue
            
        x, y = TOPOLOGY_POS[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"Узел: {node}")
        node_labels.append(node_str)

        # ЛОГИКА РАСПОЗНАВАНИЯ ТИПОВ УЗЛОВ
        if node_str.isupper() and node_str.isalpha() and len(node_str) == 1:
            # Заглавные одиночные буквы (Источники A, B, C...)
            node_marker_color.append('purple')
            node_marker_size.append(18)
            node_marker_symbol.append('square')
        elif node_str.isdigit():
            # Только цифры (Потребители 1, 2, 3...)
            node_marker_color.append('dodgerblue')
            node_marker_size.append(12)
            node_marker_symbol.append('circle')
        else:
            # Римские цифры (Крупные узлы связи I., II., III....)
            node_marker_color.append('tomato')
            node_marker_size.append(22)
            node_marker_symbol.append('circle')

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_labels,
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

    # Обновленная Легенда
    legend_items = [
        ('Источники', 'purple', 'square', 15),
        ('Потребители', 'dodgerblue', 'circle', 10),
        ('Узлы связи', 'tomato', 'circle', 15),
    ]
    for name, color, symbol, size in legend_items:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(size=size, color=color, symbol=symbol), name=name, showlegend=True))

    fig.update_layout(
        title='⚡ Интерактивная карта распределительной сети',
        title_font_size=24, dragmode='pan', hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template='plotly_white', height=800,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="Black", borderwidth=1)
    )

    return fig
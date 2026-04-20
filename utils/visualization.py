import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from core.data_loader import TOPOLOGY_POS

def draw_heatmap(nodes, adj, capacities, final_load):
    G = nx.DiGraph()
    for u in adj:
        for v in adj[u]:
            G.add_edge(u, v, capacity=capacities.get((u, v), 0))

    # --- ОПРЕДЕЛЕНИЕ КООРДИНАТ (СХЕМАТИЧНЫЙ ЛЕЙАУТ) ---
    # Мы используем централизованные координаты из data_loader.py
    pos = {}
    for node in nodes:
        if node in TOPOLOGY_POS:
            pos[node] = TOPOLOGY_POS[node]
        else:
            # Ищем, к какому узлу подключен этот потребитель
            parent = None
            for (u, v) in capacities.keys():
                if v == node:
                    parent = u
                    break
            
            if parent and parent in TOPOLOGY_POS:
                px, py = TOPOLOGY_POS[parent]
                offset = 1.0
                try:
                    val = int(node)
                    pos[node] = (px + offset * 0.5, py - offset) # Смещение вниз
                except ValueError:
                    pos[node] = (px + offset, py - offset)
            else:
                pos[node] = (0, 0)

    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_title("⚡ Тепловая карта распределения потоков (Схематичный вид)", fontsize=16, fontweight='bold', pad=20)

    sources, consumers, connectors = [], [], []
    for n in nodes:
        if str(n).isalpha() and len(str(n)) == 1:
            sources.append(n)
        elif str(n).isdigit() or (str(n).endswith('.') and str(n)[:-1].isdigit()):
            consumers.append(n)
        else:
            connectors.append(n)

    # Отрисовка узлов
    nx.draw_networkx_nodes(G, pos, nodelist=sources, node_shape='s', node_color='purple', node_size=700, edgecolors='black', ax=ax, label="Источники (A-P)")
    nx.draw_networkx_nodes(G, pos, nodelist=consumers, node_shape='o', node_color='dodgerblue', node_size=500, edgecolors='black', ax=ax, label="Потребители (1-38)")
    nx.draw_networkx_nodes(G, pos, nodelist=connectors, node_shape='o', node_color='tomato', node_size=800, edgecolors='black', ax=ax, label="Узел связи (I.-XVIII.)")

    nx.draw_networkx_labels(G, pos, font_size=9, font_color='white', font_weight='bold', ax=ax)

    cmap = plt.colormaps['coolwarm']
    edges_to_draw, edge_colors, labels = [], [], {}

    for edge, cap in capacities.items():
        load = final_load.get(edge, 0.0)
        if load >= 0.1:
            ratio = min(load / cap, 1.0) if cap > 0 else 0.0
            edges_to_draw.append(edge)
            edge_colors.append(cmap(ratio))
            labels[edge] = f"{load:.1f}"

    if edges_to_draw:
        nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, edge_color=edge_colors,
                               width=2, arrowstyle='-|>', arrowsize=15, connectionstyle="arc3,rad=0.05", ax=ax)

    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    fig.colorbar(sm, ax=ax, label='Уровень загрузки участка (0.0 - 1.0)', pad=0.02, shrink=0.6)
    
    plt.tight_layout()
    ax.axis('off')
    return fig
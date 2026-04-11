import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import math


def draw_heatmap(nodes, adj, capacities, final_load):
    G = nx.DiGraph()
    for u in adj:
        for v in adj[u]:
            G.add_edge(u, v, capacity=capacities.get((u, v), 0))

    pos = {node: (5 * math.cos(2 * math.pi * i / len(nodes)), 5 * math.sin(2 * math.pi * i / len(nodes)))
           for i, node in enumerate(nodes)}

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Тепловая карта загрузки распределительной сети", fontsize=14)

    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=800, edgecolors='black', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

    cmap = plt.colormaps['coolwarm']
    edges_to_draw, edge_colors, labels = [], [], {}

    for edge, cap in capacities.items():
        load = final_load.get(edge, 0.0)
        if load >= 1.0:
            ratio = min(load / cap, 1.0) if cap > 0 else 1.0
            edges_to_draw.append(edge)
            edge_colors.append(cmap(ratio))
            labels[edge] = f"{load:.3f}/{cap:.3f}"  # Точность до 3 знаков (Требование 9.1.2)

    if edges_to_draw:
        nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, edge_color=edge_colors,
                               width=2.5, arrowstyle='->', connectionstyle="arc3,rad=0.1", ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=7, ax=ax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    fig.colorbar(sm, ax=ax, label='Уровень загрузки участка')
    ax.axis('off')

    return fig
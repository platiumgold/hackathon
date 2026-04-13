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

    # Determine node types based on typical naming conventions from the image
    sources, consumers, connectors = [], [], []
    for n in nodes:
        if str(n).isalpha() and len(str(n)) == 1:
            sources.append(n)
        elif str(n).isdigit():
            consumers.append(n)
        else:
            connectors.append(n)

    # Draw sources as purple squares
    nx.draw_networkx_nodes(G, pos, nodelist=sources, node_shape='s', node_color='purple', node_size=800, edgecolors='black', ax=ax, label="Источник потока")
    # Draw consumers as blue circles
    nx.draw_networkx_nodes(G, pos, nodelist=consumers, node_shape='o', node_color='dodgerblue', node_size=800, edgecolors='black', ax=ax, label="Потребитель потока")
    # Draw connection nodes as red circles
    nx.draw_networkx_nodes(G, pos, nodelist=connectors, node_shape='o', node_color='tomato', node_size=800, edgecolors='black', ax=ax, label="Узел соединения потоков")

    nx.draw_networkx_labels(G, pos, font_size=10, font_color='white', font_weight='bold', ax=ax)

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
                               width=2.5, arrowstyle='-|>', arrowsize=20, connectionstyle="arc3,rad=0.1", ax=ax,
                               min_source_margin=15, min_target_margin=15)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=7, ax=ax)

    # Show Custom Legend
    ax.legend(scatterpoints=1, loc='upper right', bbox_to_anchor=(1.2, 1.0))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    fig.colorbar(sm, ax=ax, label='Уровень загрузки участка')
    ax.axis('off')

    return fig
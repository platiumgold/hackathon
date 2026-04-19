import random
import networkx as nx
import pandas as pd

def generate_and_save_csv(num_nodes=40, num_edges=120, num_reqs=30, req_filename="table_1_1_complex.csv", cap_filename="table_1_2_complex.csv"):
    """
    Генерирует сложный случайный связный граф сети и заявок, 
    а затем сохраняет их в CSV файлы, совместимые с загрузчиком (app.py).
    """
    print(f"Generating network with ~{num_nodes} nodes, {num_edges} edges and {num_reqs} requests...")
    
    # 1. Генерация графа
    G = nx.gnm_random_graph(num_nodes, num_edges, directed=True)
    
    # Берем только самую большую связную компоненту, чтобы гарантировать пути
    if not nx.is_weakly_connected(G):
        components = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
        G = G.subgraph(components[0]).copy()
        
    nodes = [str(n) for n in G.nodes()]
    
    # 2. Формирование ограничений сети (Таблица 1.2)
    cap_data = []
    for u, v in G.edges():
        u_str, v_str = str(u), str(v)
        if u_str != v_str:
            # Случайная допустимая мощность от 50 до 500 кВт
            cap = round(random.uniform(50.0, 500.0), 2)
            cap_data.append({"начало": u_str, "окончание": v_str, "Допустимая мощность": cap})
            
    # 3. Формирование заявок (Таблица 1.1)
    req_data = []
    # Чтобы избежать дубликатов заявок между одними и теми же узлами, собираем в словарь
    req_dict = {}
    for _ in range(num_reqs):
        src = random.choice(nodes)
        dst = random.choice(nodes)
        while src == dst:
            dst = random.choice(nodes)
            
        amount = round(random.uniform(20.0, 200.0), 2)
        req_dict[(src, dst)] = req_dict.get((src, dst), 0) + amount
        
    for (src, dst), amount in req_dict.items():
        req_data.append({"Источник потока": src, "Потребитель": dst, "Поток, кВт": round(amount, 2)})
        
    # 4. Сохранение в CSV
    df_req = pd.DataFrame(req_data)
    df_cap = pd.DataFrame(cap_data)
    
    # Сохраняем с разделителем запятая (стандарт) или точка с запятой
    df_req.to_csv(req_filename, index=False, sep=",")
    df_cap.to_csv(cap_filename, index=False, sep=",")
    
    total_power = sum([r["Поток, кВт"] for r in req_data])
    
    print(f"Successfully saved:")
    print(f"  - Requests: {req_filename} ({len(req_data)} rows, total power: {total_power:.2f} kW)")
    print(f"  - Network:  {cap_filename} ({len(cap_data)} edges)")

if __name__ == "__main__":
    # Фиксируем seed для повторяемости (можно убрать)
    random.seed(42)
    generate_and_save_csv(num_nodes=40, num_edges=120, num_reqs=30)

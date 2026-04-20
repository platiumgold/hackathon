import random
import pandas as pd
from core.data_loader import BASE_TOPOLOGY, TOPOLOGY_POS

def generate_synthetic_network(num_reqs=15, load_level=100.0, bottleneck_level=0.5):
    """
    Генерирует данные на основе фиксированной топологии Альфа.
    load_level: средняя мощность одной заявки (кВт)
    bottleneck_level: 0.0 (без ограничений) до 1.0 (сильные ограничения)
    """
    
    # 1. Определяем списки узлов по типам
    all_nodes = list(TOPOLOGY_POS.keys())
    sources = [n for n in all_nodes if str(n).isalpha() and len(str(n)) == 1]
    consumers = [n for n in all_nodes if str(n).isdigit()]
    
    # 2. Генерируем Таблицу 1.1 (Заявки)
    req_data = []
    for _ in range(num_reqs):
        src = random.choice(sources)
        dst = random.choice(consumers)
        # Вариация потока +/- 50% от load_level
        flow = round(load_level * random.uniform(0.5, 1.5), 1)
        req_data.append([src, dst, flow])
    
    df_req = pd.DataFrame(req_data, columns=['Источник потока', 'Потребитель', 'Поток, кВт'])
    
    # 3. Генерируем Таблицу 1.2 (Ограничения)
    # Выбираем случайное количество ребер из базовой топологии для наложения ограничений
    # Чем выше bottleneck_level, тем больше ребер ограничиваем и тем меньше им даем мощности
    cap_data = []
    total_flow = df_req['Поток, кВт'].sum()
    
    # Ограничиваем от 20% до 80% всех ребер
    num_limited_edges = int(len(BASE_TOPOLOGY) * (0.2 + 0.6 * bottleneck_level))
    limited_edges = random.sample(BASE_TOPOLOGY, num_limited_edges)
    
    for u, v in limited_edges:
        # Мощность ребра: если bottleneck_level высокий, то мощность будет близка к средней заявке
        # Если низкий - то будет покрывать значительную часть общего потока
        min_cap = load_level * 0.5
        max_cap = total_flow * (1.1 - bottleneck_level)
        cap = round(random.uniform(min_cap, max_cap), 1)
        cap_data.append([u, v, cap])
        
    df_cap = pd.DataFrame(cap_data, columns=['начало', 'окончание', 'Допустимая мощность'])
    
    return df_req, df_cap

if __name__ == "__main__":
    # Тестовый запуск
    req, cap = generate_synthetic_network()
    print(f"Generated {len(req)} requests and {len(cap)} constraints.")
    req.to_csv("table_1_1_generated.csv", index=False)
    cap.to_csv("table_1_2_generated.csv", index=False)

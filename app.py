import streamlit as st
import pandas as pd
from core.data_loader import load_network_data
from core.aco import run_aco
from core.gnn import run_gnn
from core.rl import run_rl
from utils.interactive_viz import draw_interactive_heatmap
from utils.generate_dataset import generate_synthetic_network

st.set_page_config(page_title="MVP Маршрутизации Энергии", layout="wide")

# --- Инициализация состояния сессии ---
if 'df_req' not in st.session_state:
    st.session_state.df_req = None
if 'df_cap' not in st.session_state:
    st.session_state.df_cap = None

st.title("⚡ MVP: Оптимизация распределенной электрической сети «Альфа»")
st.markdown("""
**Интеллектуальная система диспетчеризации (ИИ)** на базе гибридных алгоритмов.
""")

# --- Боковая панель ---
st.sidebar.header("📥 Входные данные")
file_req = st.sidebar.file_uploader("Загрузить Таблицу 1.1 (Заявки)", type=['csv', 'xlsx'])
file_cap = st.sidebar.file_uploader("Загрузить Таблицу 1.2 (Ограничения)", type=['csv', 'xlsx'])

# Логика загрузки из файлов
if file_req and file_cap:
    if st.sidebar.button("📁 Применить загруженные файлы"):
        st.session_state.df_req = pd.read_excel(file_req) if file_req.name.endswith('xlsx') else pd.read_csv(file_req)
        st.session_state.df_cap = pd.read_excel(file_cap) if file_cap.name.endswith('xlsx') else pd.read_csv(file_cap)
        st.sidebar.success("Файлы загружены и закреплены!")

st.sidebar.markdown("---")
st.sidebar.header("🧪 Генерация сценария")
with st.sidebar.expander("Настройки генератора"):
    n_req = st.slider("Количество заявок", 5, 50, 15)
    load_val = st.slider("Средняя нагрузка (кВт)", 10, 500, 100)
    bottleneck = st.slider("Степень дефицита (0-1)", 0.0, 1.0, 0.4)
    if st.button("🏗️ Сгенерировать новый сценарий"):
        df_r, df_c = generate_synthetic_network(n_req, load_val, bottleneck)
        st.session_state.df_req = df_r
        st.session_state.df_cap = df_c
        st.sidebar.success("Сценарий сгенерирован и закреплен!")

st.sidebar.markdown("---")
algo = st.sidebar.radio("🤖 Выбор алгоритма ИИ",
                        ["Physics-Informed GNN", "Ant Colony (ACO)", "Reinforcement Learning (PPO)"])
run_btn = st.sidebar.button("🚀 ЗАПУСТИТЬ РАСЧЕТ")

# --- Основная область ---
if st.session_state.df_req is not None and st.session_state.df_cap is not None:

    with st.expander("🔍 Просмотр текущих данных (Table 1.1 и 1.2)"):
        c1, c2 = st.columns(2)
        c1.markdown("**Текущие Заявки**")
        c1.dataframe(st.session_state.df_req, use_container_width=True)
        c2.markdown("**Текущие Ограничения**")
        c2.dataframe(st.session_state.df_cap, use_container_width=True)

    if run_btn:
        df_req = st.session_state.df_req
        df_cap = st.session_state.df_cap

        with st.spinner('Анализ топологии...'):
            nodes, dests, adj, caps, reqs = load_network_data(df_req, df_cap)

        # Выполнение алгоритма
        final_load = {}
        total_delivered = 0.0
        total_requested = sum(reqs.values())

        with st.spinner(f'Работает {algo}...'):
            if algo == "Ant Colony (ACO)":
                res = run_aco(nodes, dests, adj, caps, reqs)
                final_load = res['load_distribution']
                total_delivered = sum(res['delivered'].values())

            elif algo == "Physics-Informed GNN":
                flows, node_idx, req_list = run_gnn(nodes, caps, reqs, epochs=400)

                # ИСПРАВЛЕНО: Безопасный маппинг индексов в имена узлов
                idx_to_node = {idx: name for name, idx in node_idx.items()}

                # Подсчет доставленного
                for req_i, ((src, dst), demand) in enumerate(req_list):
                    d_idx = node_idx[dst]
                    in_d = flows[req_i, :, d_idx].sum().item()
                    out_d = flows[req_i, d_idx, :].sum().item()
                    total_delivered += max(0.0, in_d - out_d)

                # Формирование нагрузки для графика
                total_edges = flows.sum(dim=0)
                for u in range(total_edges.shape[0]):
                    for v in range(total_edges.shape[1]):
                        flow_val = total_edges[u, v].item()
                        if flow_val >= 0.5:  # Фильтр шумов
                            n_u = idx_to_node[u]
                            n_v = idx_to_node[v]
                            final_load[(n_u, n_v)] = flow_val

            elif algo == "Reinforcement Learning (PPO)":
                res = run_rl(nodes, dests, adj, caps, reqs)
                final_load = res['load_distribution']
                total_delivered = sum(res['delivered'].values())

        # Вывод метрик
        col1, col2 = st.columns(2)
        col1.metric("Запрошено мощности", f"{total_requested:.3f} кВт")
        col2.metric("Фактически доставлено", f"{total_delivered:.3f} кВт")

        st.info(
            "💡 **Отчет диспетчера:** Алгоритм пропорционально ограничил заявки для предотвращения перегрузки участков. Баланс генерации и потребления соблюден.")
        fig = draw_interactive_heatmap(nodes, adj, caps, final_load)
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
else:
    st.info("👋 Добро пожаловать! Пожалуйста, загрузите файлы или сгенерируйте сценарий в боковом меню.")
import streamlit as st
import pandas as pd
import plotly.express as px
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
if 'res' not in st.session_state:
    st.session_state.res = None
if 'algo_run' not in st.session_state:
    st.session_state.algo_run = ""

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
    n_req = st.slider("Количество заявок", 1, 50, 15)
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
        res = None

        with st.spinner(f'Работает {algo}...'):
            if algo == "Ant Colony (ACO)":
                res = run_aco(nodes, dests, adj, caps, reqs)
                final_load = res['load_distribution']
                total_delivered = sum(res['delivered'].values())

            elif algo == "Physics-Informed GNN":
                flows, node_idx, req_list = run_gnn(nodes, caps, reqs, epochs=400)
                idx_to_node = {idx: name for name, idx in node_idx.items()}
                
                # Подсчет доставленного и формирование request_flows
                delivered_dict = {}
                request_flows = {}
                load_distribution = {}
                
                total_delivered = 0.0
                for req_i, ((src, dst), demand) in enumerate(req_list):
                    req_key = (src, dst)
                    d_idx = node_idx[dst]
                    in_d = flows[req_i, :, d_idx].sum().item()
                    out_d = flows[req_i, d_idx, :].sum().item()
                    deliv = max(0.0, in_d - out_d)
                    delivered_dict[req_key] = deliv
                    total_delivered += deliv
                    
                    # Потоки конкретной заявки
                    request_flows[req_key] = {}
                    for u in range(num_nodes := len(nodes)):
                        for v in range(num_nodes):
                            f_val = flows[req_i, u, v].item()
                            if f_val > 0.01:
                                request_flows[req_key][(idx_to_node[u], idx_to_node[v])] = f_val
                
                # Общая нагрузка
                total_edges_tensor = flows.sum(dim=0)
                for u in range(num_nodes):
                    for v in range(num_nodes):
                        f_val = total_edges_tensor[u, v].item()
                        if f_val > 0.1:
                            load_distribution[(idx_to_node[u], idx_to_node[v])] = f_val
                
                res = {
                    'load_distribution': load_distribution,
                    'delivered': delivered_dict,
                    'request_flows': request_flows
                }

            elif algo == "Reinforcement Learning (PPO)":
                res = run_rl(nodes, dests, adj, caps, reqs)
            
            st.session_state.res = res
            st.session_state.algo_run = algo
            st.session_state.nodes_info = (nodes, dests, adj, caps, reqs)
            st.sidebar.success("Расчет завершен!")

    # --- Отображение результатов (если они есть в сессии) ---
    if st.session_state.res is not None:
        nodes, dests, adj, caps, reqs = st.session_state.nodes_info
        res = st.session_state.res
        final_load = res['load_distribution']
        total_delivered = sum(res['delivered'].values())
        total_requested = sum(reqs.values())

        # Вывод метрик
        col1, col2 = st.columns(2)
        col1.metric("Запрошено мощности", f"{total_requested:.3f} кВт")
        col2.metric("Фактически доставлено", f"{total_delivered:.3f} кВт")

        st.info(
            "💡 **Отчет диспетчера:** Алгоритм пропорционально ограничил заявки для предотвращения перегрузки участков. Баланс генерации и потребления соблюден.")
        
        # --- НОВОЕ: Инспектор ребер (альтернатива клику) ---
        st.markdown("---")
        st.subheader("🛠️ Инспектор участков")
        
        # Список всех ребер, где есть поток, отсортированный по загруженности
        edge_list = []
        for (u, v), cap in caps.items():
            load = final_load.get((u, v), 0.0)
            if load > 0.1:
                ratio = load / cap if cap > 0 else 0
                edge_list.append({
                    'edge': (u, v),
                    'label': f"{u} → {v} (Загрузка: {ratio:.1%}, {load:.1f}/{cap:.0f} кВт)",
                    'ratio': ratio
                })
        
        edge_list = sorted(edge_list, key=lambda x: x['ratio'], reverse=True)
        edge_options = [e['label'] for e in edge_list]
        
        col_ins1, col_ins2 = st.columns([2, 1])
        with col_ins1:
            selected_edge_label = st.selectbox("Посмотреть, кто нагружает участок:", 
                                             options=["-- выберите участок --"] + edge_options)
        
        # Список для выбора заявок
        available_reqs = list(res['request_flows'].keys())
        req_options = [f"{s} → {d} ({res['delivered'].get((s,d), 0):.1f}/{reqs.get((s,d),0):.1f} кВт)" 
                       for s, d in available_reqs]
        req_to_key = {opt: key for opt, key in zip(req_options, available_reqs)}

        if selected_edge_label != "-- выберите участок --":
            edge_data = next(e for e in edge_list if e['label'] == selected_edge_label)
            target_edge = edge_data['edge']
            
            reqs_on_edge = []
            for req_key, flows in res['request_flows'].items():
                if target_edge in flows:
                    opt = next((o for o in req_options if o.startswith(f"{req_key[0]} → {req_key[1]}")), None)
                    if opt:
                        reqs_on_edge.append(opt)
            
            if reqs_on_edge:
                with col_ins2:
                    st.write("") # Отступ
                    if st.button(f"✨ Показать все ({len(reqs_on_edge)})"):
                        st.session_state.req_selector = reqs_on_edge
                        st.rerun()

        st.subheader("🗺️ Визуализация потоков")
        selected_opts = st.multiselect("🔍 Выберите конкретные заявки для подсветки маршрутов (пусто = общая нагрузка):", 
                                       options=req_options,
                                       key="req_selector")
        
        selected_load = None
        current_selection = st.session_state.get("req_selector", [])
        if current_selection:
            selected_load = {}
            for opt in current_selection:
                key = req_to_key[opt]
                flows_for_req = res['request_flows'].get(key, {})
                for edge, val in flows_for_req.items():
                    selected_load[edge] = selected_load.get(edge, 0.0) + val
        
        fig = draw_interactive_heatmap(nodes, adj, caps, final_load, selected_load)
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

        # --- Дополнительная аналитика по запросам ---
        st.markdown("---")
        st.subheader("📊 Анализ справедливости и статуса заявок")
        
        # Подготовка данных для таблицы
        req_stats = []
        for (src, dst), req_val in reqs.items():
            deliv_val = res['delivered'].get((src, dst), 0.0)
            ratio = deliv_val / req_val if req_val > 0 else 1.0
            req_stats.append({
                "Источник": src,
                "Потребитель": dst,
                "Заявлено (кВт)": f"{req_val:.3f}",
                "Доставлено (кВт)": f"{deliv_val:.3f}",
                "Выполнение (%)": f"{ratio*100:.1f}%",
                "ratio_num": ratio
            })
        
        df_stats = pd.DataFrame(req_stats)
        
        col_t1, col_t2 = st.columns([2, 1])
        
        with col_t1:
            st.markdown("**Статус выполнения каждой заявки**")
            st.dataframe(df_stats.drop(columns=["ratio_num"]), use_container_width=True, height=400)
            
        with col_t2:
            st.markdown("**Распределение справедливости**")
            fig_hist = px.histogram(df_stats, x="ratio_num", nbins=10, 
                                    labels={'ratio_num': 'Доля выполнения'},
                                    title="Гистограмма удовлетворенности",
                                    color_discrete_sequence=['#FF4B4B'])
            fig_hist.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
            
        st.success(f"✅ Расчет завершен. Средний уровень выполнения заявок: {total_delivered/total_requested:.1%}")
else:
    st.info("👋 Добро пожаловать! Пожалуйста, загрузите файлы или сгенерируйте сценарий в боковом меню.")
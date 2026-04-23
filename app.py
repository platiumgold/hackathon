import streamlit as st
import pandas as pd
import plotly.express as px
import pdfplumber
from core.data_loader import load_network_data, get_logical_edges, get_current_topology
from core.aco import run_aco
from core.gnn import run_gnn
from core.rl import run_rl
from utils.interactive_viz import draw_interactive_heatmap
from utils.generate_dataset import generate_synthetic_network

st.set_page_config(page_title="MVP Маршрутизации Энергии", layout="wide")


# --- Вспомогательные функции ---
def reset_results():
    st.session_state.res = None


def toggle_request(opt_label):
    current = st.session_state.get("req_selector", [])
    if opt_label in current:
        current.remove(opt_label)
    else:
        current.append(opt_label)
    st.session_state.req_selector = current


def clean_value(val):
    if not val: return 0.0
    try:
        return float(str(val).replace(' ', '').replace(',', '.'))
    except ValueError:
        return 0.0


def process_pdf_req(file):
    all_rows = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table: all_rows.extend(table)
    cleaned = []
    for row in all_rows:
        if not row or len(row) < 4: continue
        src = str(row[1]).strip() if row[1] else ""
        dst = str(row[2]).strip() if row[2] else ""
        if "Источник" in src or "Потребитель" in dst or not src or not dst: continue
        if dst.lower() == "итого" or "всего" in src.lower(): continue
        val = clean_value(row[3])
        cleaned.append([src, dst, val])
    return pd.DataFrame(cleaned, columns=["Источник потока", "Потребитель", "Поток, кВт"])


def process_pdf_cap(file):
    all_rows = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table: all_rows.extend(table)
    cleaned = []
    for row in all_rows:
        if not row or len(row) < 4: continue
        u = str(row[1]).strip() if row[1] else ""
        v = str(row[2]).strip() if row[2] else ""
        if "начало" in u or "окончание" in v or not u or not v: continue
        if row[0] == "1" and row[1] == "2" and row[2] == "3": continue
        cap = clean_value(row[3])
        if cap > 0:
            cleaned.append([u, v, cap])
    return pd.DataFrame(cleaned, columns=["начало", "окончание", "Допустимая мощность"])


# ПУНКТ 2: Инициализация состояния сессии со стандартными данными
if 'df_req' not in st.session_state or 'df_cap' not in st.session_state:
    r, c = generate_synthetic_network(15, 100.0, 0.4)
    if 'df_req' not in st.session_state: st.session_state.df_req = r
    if 'df_cap' not in st.session_state: st.session_state.df_cap = c

if 'res' not in st.session_state:
    st.session_state.res = None
if 'algo_run' not in st.session_state:
    st.session_state.algo_run = ""
if 'reversed_logical_edges' not in st.session_state:
    st.session_state.reversed_logical_edges = []
if 'potentially_reversible' not in st.session_state:
    st.session_state.potentially_reversible = [('VI.', 'V.')]
if 'nodes_info' not in st.session_state:
    st.session_state.nodes_info = ([], [], {}, {}, {}, [])

st.title("⚡ MVP: Оптимизация распределенной электрической сети «Альфа»")
st.markdown("**Интеллектуальная система диспетчеризации (ИИ)** на базе гибридных алгоритмов.")

# --- Боковая панель ---
st.sidebar.header("📥 Входные данные")
file_req = st.sidebar.file_uploader("Загрузить Заявки (CSV/Excel/PDF)", type=['csv', 'xlsx', 'pdf'])
file_cap = st.sidebar.file_uploader("Загрузить Ограничения (CSV/Excel/PDF)", type=['csv', 'xlsx', 'pdf'])

if st.sidebar.button("📁 Применить загруженные файлы"):
    # Обрабатываем только те, что загружены
    if file_req:
        if file_req.name.endswith('.pdf'):
            st.session_state.df_req = process_pdf_req(file_req)
        elif file_req.name.endswith('.xlsx'):
            st.session_state.df_req = pd.read_excel(file_req)
        else:
            st.session_state.df_req = pd.read_csv(file_req)

    if file_cap:
        if file_cap.name.endswith('.pdf'):
            st.session_state.df_cap = process_pdf_cap(file_cap)
        elif file_cap.name.endswith('.xlsx'):
            st.session_state.df_cap = pd.read_excel(file_cap)
        else:
            st.session_state.df_cap = pd.read_csv(file_cap)

    reset_results()
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
        reset_results()
        st.sidebar.success("Сценарий сгенерирован!")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Пульт управления реверсом")
# ПУНКТ 4: Убрана возможность добавлять ребра
for edge in st.session_state.potentially_reversible:
    u, v = edge
    is_active = edge in st.session_state.reversed_logical_edges
    label = f"{'🔄' if is_active else '➡️'} {u} ↔ {v}"
    if st.sidebar.button(label, key=f"btn_rev_{u}_{v}", use_container_width=True):
        if is_active:
            st.session_state.reversed_logical_edges.remove(edge)
        else:
            st.session_state.reversed_logical_edges.append(edge)
        reset_results()
        st.rerun()

st.sidebar.markdown("---")
algo = st.sidebar.radio("🤖 Выбор алгоритма ИИ",
                        ["Physics-Informed GNN", "Ant Colony (ACO)", "Reinforcement Learning (PPO)"])

# ПУНКТ 5: Настройки алгоритмов
with st.sidebar.expander("🛠 Настройки обучения"):
    if algo == "Ant Colony (ACO)":
        epochs = st.number_input("Кол-во итераций", min_value=10, max_value=500, value=30)
    else:
        epochs = st.number_input("Максимум эпох", min_value=10, max_value=1000, value=150)

    time_limit = st.slider("Ограничение по времени (мин)", 1, 60, 5)
    use_early_stop = st.checkbox("Ранняя остановка (Early Stop)", value=True,
                                 help="Остановить обучение, если метрики перестали улучшаться.")

run_btn = st.sidebar.button("🚀 ЗАПУСТИТЬ РАСЧЕТ", type="primary", use_container_width=True)

# --- ГЛОБАЛЬНАЯ ПОДГОТОВКА ТОПОЛОГИИ ---
current_topo = get_current_topology(st.session_state.reversed_logical_edges)
# Получаем 6 значений, включая список невалидных заявок
nodes, dests, adj, caps, reqs, invalid_reqs = load_network_data(st.session_state.df_req, st.session_state.df_cap,
                                                                current_topo)
st.session_state.nodes_info = (nodes, dests, adj, caps, reqs, invalid_reqs)

# ПУНКТ 6: Вывод уведомления об отсутствии путей
if invalid_reqs:
    st.warning(f"⚠️ Внимание! Обнаружено {len(invalid_reqs)} заявок без физического пути. Они исключены из обучения.")
    with st.expander("Посмотреть недостижимые заявки"):
        st.dataframe(pd.DataFrame(invalid_reqs))

# --- Основная область ---
if st.session_state.df_req is not None and st.session_state.df_cap is not None:

    st.subheader("📝 Масштабирование и ручное редактирование данных")
    st.info("💡 **Вы можете менять числа прямо в таблицах ниже.** Все ваши правки будут учтены при запуске.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Таблица 1.1: Заявки (Редактируемо)**")
        edited_req = st.data_editor(st.session_state.df_req, num_rows="dynamic", use_container_width=True,
                                    on_change=reset_results)
    with c2:
        st.markdown("**Таблица 1.2: Доп. мощности (Редактируемо)**")
        edited_cap = st.data_editor(st.session_state.df_cap, num_rows="dynamic", use_container_width=True,
                                    on_change=reset_results)

    st.session_state.df_req = edited_req
    st.session_state.df_cap = edited_cap

    if run_btn:
        res = None
        with st.spinner(f'Работает {algo}... Пожалуйста, подождите.'):
            if algo == "Ant Colony (ACO)":
                res = run_aco(nodes, dests, adj, caps, reqs, n_iterations=epochs, time_limit_min=time_limit,
                              early_stop=use_early_stop)
            elif algo == "Physics-Informed GNN":
                res = run_gnn(nodes, caps, reqs, epochs=epochs, time_limit_min=time_limit, early_stop=use_early_stop)
            elif algo == "Reinforcement Learning (PPO)":
                res = run_rl(nodes, dests, adj, caps, reqs, epochs=epochs, time_limit_min=time_limit,
                             early_stop=use_early_stop)

            st.session_state.res = res
            st.session_state.algo_run = algo
            st.sidebar.success("Расчет завершен!")

    if st.session_state.res is not None:
        nodes, dests, adj, caps, reqs, _ = st.session_state.nodes_info
        res = st.session_state.res
        final_load = res.get('load_distribution', {})
        total_delivered = sum(res.get('delivered', {}).values())
        total_requested = sum(reqs.values())

        col1, col2 = st.columns(2)
        col1.metric("Запрошено мощности", f"{total_requested:.3f} кВт")
        col2.metric("Фактически доставлено", f"{total_delivered:.3f} кВт")

        st.info(
            "💡 **Отчет диспетчера:** Алгоритм пропорционально ограничил заявки для предотвращения перегрузки участков.")

        st.markdown("---")
        st.subheader("🛠️ Инспектор участков")

        edge_list = []
        for (u, v), cap in caps.items():
            load = final_load.get((u, v), 0.0)
            if load > 0.1:
                ratio = load / cap if cap > 0 else 0
                edge_list.append({
                    'edge': (u, v),
                    'label': f"{u} → {v} (Загрузка: {ratio:.1%}, {load:.3f}/{cap:.0f} кВт)",
                    'ratio': ratio
                })

        edge_list = sorted(edge_list, key=lambda x: x['ratio'], reverse=True)
        edge_options = [e['label'] for e in edge_list]

        col_ins1, col_ins2 = st.columns([2, 1])
        with col_ins1:
            selected_edge_label = st.selectbox("Посмотреть, кто нагружает участок:",
                                               options=["-- выберите участок --"] + edge_options)

        available_reqs = list(res['request_flows'].keys())
        req_to_opt = {}
        opt_to_key = {}
        for r_key in available_reqs:
            s, d = r_key
            deliv = res['delivered'].get(r_key, 0.0)
            reqv = reqs.get(r_key, 0.0)
            opt_str = f"{s} → {d} ({deliv:.3f}/{reqv:.3f} кВт)"
            req_to_opt[r_key] = opt_str
            opt_to_key[opt_str] = r_key

        req_options = list(opt_to_key.keys())

        if selected_edge_label != "-- выберите участок --":
            edge_data = next((e for e in edge_list if e['label'] == selected_edge_label), None)
            if edge_data:
                target_edge = edge_data['edge']
                breakdown_data = []
                for r_key, flows in res['request_flows'].items():
                    flow_on_this_edge = flows.get(target_edge, 0.0)
                    if flow_on_this_edge > 0.001:
                        s, d = r_key
                        breakdown_data.append({"key": r_key, "Заявка": f"{s} → {d}", "flow": flow_on_this_edge})

                if breakdown_data:
                    with col_ins1:
                        with st.expander(f"⚙️ Управление потоками участка: {selected_edge_label}", expanded=True):
                            st.write("Выберите заявки для отображения маршрутов:")
                            current_selection = st.session_state.get("req_selector", [])
                            for item in breakdown_data:
                                r_key = item['key']
                                flow_val = item['flow']
                                opt_label = req_to_opt[r_key]
                                is_checked = opt_label in current_selection
                                cb_label = f"{item['Заявка']} | **Вклад: {flow_val:.3f} кВт**"
                                st.checkbox(cb_label, value=is_checked, key=f"cb_{r_key[0]}_{r_key[1]}",
                                            on_change=toggle_request, args=(opt_label,))

                    with col_ins2:
                        if st.button("✨ Подсветить все на участке", use_container_width=True):
                            new_sel = list(set(current_selection + [req_to_opt[i['key']] for i in breakdown_data]))
                            st.session_state.req_selector = new_sel
                            st.rerun()
                        if st.button("🧹 Скрыть все на участке", use_container_width=True):
                            new_sel = [s for s in current_selection if
                                       s not in [req_to_opt[i['key']] for i in breakdown_data]]
                            st.session_state.req_selector = new_sel
                            st.rerun()

        st.subheader("🗺️ Визуализация потоков")
        selected_opts = st.multiselect("🔍 Выберите заявки (пусто = общая нагрузка):",
                                       options=req_options, key="req_selector")

        selected_load = None
        current_selection = st.session_state.get("req_selector", [])
        if current_selection:
            selected_load = {}
            for opt in current_selection:
                if opt in opt_to_key:
                    r_key = opt_to_key[opt]
                    flows_for_req = res['request_flows'].get(r_key, {})
                    for edge, val in flows_for_req.items():
                        selected_load[edge] = selected_load.get(edge, 0.0) + val

        fig = draw_interactive_heatmap(nodes, adj, caps, final_load, selected_load)
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

        st.markdown("---")
        st.subheader("📊 Анализ справедливости")
        req_stats = []
        for (src, dst), req_val in reqs.items():
            deliv_val = res['delivered'].get((src, dst), 0.0)
            ratio = deliv_val / req_val if req_val > 0 else 1.0
            req_stats.append({"Источник": src, "Потребитель": dst, "Заявлено (кВт)": f"{req_val:.3f}",
                              "Доставлено (кВт)": f"{deliv_val:.3f}", "Выполнение (%)": f"{ratio * 100:.1f}%",
                              "ratio_num": ratio})
        df_stats = pd.DataFrame(req_stats)
        col_t1, col_t2 = st.columns([2, 1])
        with col_t1:
            st.dataframe(df_stats.drop(columns=["ratio_num"]), use_container_width=True, height=400)
        with col_t2:
            fig_hist = px.histogram(df_stats, x="ratio_num", nbins=10, labels={'ratio_num': 'Доля выполнения'},
                                    title="Гистограмма удовлетворенности", color_discrete_sequence=['#FF4B4B'])
            st.plotly_chart(fig_hist, use_container_width=True)

    if st.session_state.res is None:
        st.subheader("🗺️ Базовая топология сети")
        st.info("Для расчета потоков нажмите «Запустить расчет».")
        fig = draw_interactive_heatmap(nodes, adj, caps, {})
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
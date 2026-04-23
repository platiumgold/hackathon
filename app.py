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


# --- Вспомогательные функции для парсинга PDF ---
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


# --- Инициализация состояния сессии ---
if 'df_req' not in st.session_state:
    st.session_state.df_req = None
if 'df_cap' not in st.session_state:
    st.session_state.df_cap = None
if 'res' not in st.session_state:
    st.session_state.res = None
if 'algo_run' not in st.session_state:
    st.session_state.algo_run = ""
if 'reversed_logical_edges' not in st.session_state:
    st.session_state.reversed_logical_edges = []
if 'potentially_reversible' not in st.session_state:
    st.session_state.potentially_reversible = [('VI.', 'V.')]

st.title("⚡ MVP: Оптимизация распределенной электрической сети «Альфа»")
st.markdown("**Интеллектуальная система диспетчеризации (ИИ)** на базе гибридных алгоритмов.")

# --- Боковая панель ---
st.sidebar.header("📥 Входные данные")
file_req = st.sidebar.file_uploader("Загрузить Заявки (CSV/Excel/PDF)", type=['csv', 'xlsx', 'pdf'])
file_cap = st.sidebar.file_uploader("Загрузить Ограничения (CSV/Excel/PDF)", type=['csv', 'xlsx', 'pdf'])

if file_req and file_cap:
    if st.sidebar.button("📁 Применить загруженные файлы"):
        # Парсинг Заявок
        if file_req.name.endswith('.pdf'):
            st.session_state.df_req = process_pdf_req(file_req)
        elif file_req.name.endswith('.xlsx'):
            st.session_state.df_req = pd.read_excel(file_req)
        else:
            st.session_state.df_req = pd.read_csv(file_req)

        # Парсинг Ограничений
        if file_cap.name.endswith('.pdf'):
            st.session_state.df_cap = process_pdf_cap(file_cap)
        elif file_cap.name.endswith('.xlsx'):
            st.session_state.df_cap = pd.read_excel(file_cap)
        else:
            st.session_state.df_cap = pd.read_csv(file_cap)

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
st.sidebar.header("⚙️ Пульт управления реверсом")

# 1. Список активных кнопок для реверса
for edge in st.session_state.potentially_reversible:
    u, v = edge
    is_active = edge in st.session_state.reversed_logical_edges
    
    # Красивая кнопка-переключатель
    label = f"{'🔄' if is_active else '➡️'} {u} ↔ {v}"
    if st.sidebar.button(label, key=f"btn_rev_{u}_{v}", use_container_width=True):
        if is_active:
            st.session_state.reversed_logical_edges.remove(edge)
        else:
            st.session_state.reversed_logical_edges.append(edge)
        st.rerun()

# 2. Добавление новых участков в пульт
st.sidebar.markdown("<br>", unsafe_allow_html=True)
all_logical = get_logical_edges()
# Исключаем те, что уже в пульте
available_to_add = [f"{u} → {v}" for u, v in all_logical 
                    if (u, v) not in st.session_state.potentially_reversible]

with st.sidebar.expander("➕ Добавить участок в пульт"):
    new_edge_str = st.selectbox("Выберите участок:", options=available_to_add, key="add_rev_select")
    if st.button("Добавить в список управления", use_container_width=True):
        if new_edge_str:
            u_n, v_n = new_edge_str.split(" → ")
            st.session_state.potentially_reversible.append((u_n, v_n))
            st.rerun()

algo = st.sidebar.radio("🤖 Выбор алгоритма ИИ",
                        ["Physics-Informed GNN", "Ant Colony (ACO)", "Reinforcement Learning (PPO)"])

run_btn = st.sidebar.button("🚀 ЗАПУСТИТЬ РАСЧЕТ", type="primary", use_container_width=True)

# --- ГЛОБАЛЬНАЯ ПОДГОТОВКА ТОПОЛОГИИ ---
current_topo = get_current_topology(st.session_state.reversed_logical_edges)
nodes, dests, adj, caps, reqs = load_network_data(st.session_state.df_req, st.session_state.df_cap, current_topo)
st.session_state.nodes_info = (nodes, dests, adj, caps, reqs)

# --- Основная область: Редактирование ---
if st.session_state.df_req is not None and st.session_state.df_cap is not None:

    st.subheader("📝 Масштабирование и ручное редактирование данных")
    st.info(
        "💡 **Вы можете менять числа, добавлять или удалять строки прямо в таблицах ниже.** Нажмите на ячейку для изменения. Чтобы добавить новую связь, пролистайте вниз таблицы. Все ваши правки будут учтены при нажатии кнопки «Запустить расчет».")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Таблица 1.1: Заявки (Редактируемо)**")
        edited_req = st.data_editor(st.session_state.df_req, num_rows="dynamic", use_container_width=True,
                                    key="editor_req")
    with c2:
        st.markdown("**Таблица 1.2: Доп. мощности (Редактируемо)**")
        edited_cap = st.data_editor(st.session_state.df_cap, num_rows="dynamic", use_container_width=True,
                                    key="editor_cap")

    # Перезаписываем данные в сессии на случай, если пользователь их отредактировал
    st.session_state.df_req = edited_req
    st.session_state.df_cap = edited_cap

    if run_btn:
        df_req = st.session_state.df_req
        df_cap = st.session_state.df_cap
        
        # Данные уже загружены в глобальном блоке выше
        nodes, dests, adj, caps, reqs = st.session_state.nodes_info

        res = None

        with st.spinner(f'Работает {algo}...'):
            if algo == "Ant Colony (ACO)":
                res = run_aco(nodes, dests, adj, caps, reqs)
            elif algo == "Physics-Informed GNN":
                # ИСПРАВЛЕНО: Передаем caps и reqs. GNN уже возвращает нужный словарь.
                res = run_gnn(nodes, caps, reqs, epochs=250)
            elif algo == "Reinforcement Learning (PPO)":
                res = run_rl(nodes, dests, adj, caps, reqs)

            st.session_state.res = res
            st.session_state.algo_run = algo
            st.session_state.nodes_info = (nodes, dests, adj, caps, reqs)
            st.sidebar.success("Расчет завершен!")

    # --- Отображение результатов ---
    if st.session_state.res is not None:
        nodes, dests, adj, caps, reqs = st.session_state.nodes_info
        res = st.session_state.res
        final_load = res.get('load_distribution', {})
        total_delivered = sum(res.get('delivered', {}).values())
        total_requested = sum(reqs.values())

        col1, col2 = st.columns(2)
        col1.metric("Запрошено мощности", f"{total_requested:.3f} кВт")
        col2.metric("Фактически доставлено", f"{total_delivered:.3f} кВт")

        st.info(
            "💡 **Отчет диспетчера:** Алгоритм пропорционально ограничил заявки для предотвращения перегрузки участков. Баланс генерации и потребления соблюден.")

        st.markdown("---")
        st.subheader("🛠️ Инспектор участков")

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

        available_reqs = list(res.get('request_flows', {}).keys())
        req_options = [f"{s} → {d} ({res['delivered'].get((s, d), 0):.1f}/{reqs.get((s, d), 0):.1f} кВт)"
                       for s, d in available_reqs]
        req_to_key = {opt: key for opt, key in zip(req_options, available_reqs)}

        if selected_edge_label != "-- выберите участок --":
            edge_data = next(e for e in edge_list if e['label'] == selected_edge_label)
            target_edge = edge_data['edge']

            reqs_on_edge = []
            for req_key, flows in res.get('request_flows', {}).items():
                if target_edge in flows:
                    opt = next((o for o in req_options if o.startswith(f"{req_key[0]} → {req_key[1]}")), None)
                    if opt:
                        reqs_on_edge.append(opt)

            if reqs_on_edge:
                with col_ins2:
                    st.write("")
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

        st.markdown("---")
        st.subheader("📊 Анализ справедливости и статуса заявок")

        req_stats = []
        for (src, dst), req_val in reqs.items():
            deliv_val = res['delivered'].get((src, dst), 0.0)
            ratio = deliv_val / req_val if req_val > 0 else 1.0
            req_stats.append({
                "Источник": src,
                "Потребитель": dst,
                "Заявлено (кВт)": f"{req_val:.3f}",
                "Доставлено (кВт)": f"{deliv_val:.3f}",
                "Выполнение (%)": f"{ratio * 100:.1f}%",
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

        st.success(
            f"✅ Расчет завершен. Средний уровень выполнения заявок: {total_delivered / (total_requested + 1e-9):.1%}")

    # --- ВСЕГДА ПОКАЗЫВАЕМ КАРТУ (даже если расчет не запущен) ---
    if st.session_state.res is None:
        st.subheader("🗺️ Визуализация топологии")
        st.info("Это текущая структура сети. Вы можете разворачивать участки в боковом меню. Для расчета потоков нажмите «Запустить расчет».")
        fig = draw_interactive_heatmap(nodes, adj, caps, {})
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
else:
    # Если данных нет, все равно показываем пустую топологию
    st.subheader("🗺️ Базовая топология сети «Альфа»")
    nodes, dests, adj, caps, reqs = st.session_state.nodes_info
    fig = draw_interactive_heatmap(nodes, adj, caps, {})
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
    st.info("👋 Пожалуйста, загрузите файлы или сгенерируйте сценарий в боковом меню для анализа нагрузок.")
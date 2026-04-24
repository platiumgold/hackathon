import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import pdfplumber
import re
from typing import List, Dict, Tuple, Any, Optional, Union
from core.data_loader import load_network_data, get_logical_edges, get_current_topology, normalize_node_strictly, parse_float_safely
from core.aco import run_aco
from core.gnn import run_gnn
from core.rl import run_rl
from utils.interactive_viz import draw_interactive_heatmap
from utils.generate_dataset import generate_synthetic_network
# Настройка Plotly: Streamlit автоматически обрабатывает офлайн-отрисовку через st.plotly_chart
# (принудительная установка рендерера удалена во избежание конфликтов в WSL)

st.set_page_config(page_title="MVP Маршрутизации Энергии", layout="wide")

# Вспомогательные функции приложения
def reset_results() -> None:
    """Сбрасывает кэшированные результаты расчетов в сессии."""
    st.session_state.res = None


def toggle_request(opt_label: str) -> None:
    """Переключает выбор заявки в мультиселекторе."""
    current = st.session_state.get("req_selector", [])
    if opt_label in current:
        current.remove(opt_label)
    else:
        current.append(opt_label)
    st.session_state.req_selector = current


def process_pdf_req(file: Any) -> pd.DataFrame:
    """
    Парсит PDF-файл с таблицей заявок с защитой от шума.
    """
    all_rows = []
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if table: all_rows.extend(table)
    except Exception as e:
        st.error(f"Ошибка чтения PDF: {e}")
        return pd.DataFrame(columns=["Источник потока", "Потребитель", "Поток, кВт"])

    cleaned = []
    for row in all_rows:
        if not row or len(row) < 4: continue
        src = normalize_node_strictly(row[1])
        dst = normalize_node_strictly(row[2])
        if "Источник" in src or "Потребитель" in dst or not src or not dst: continue
        if dst.lower() == "итого" or "всего" in src.lower(): continue
        val = parse_float_safely(row[3])
        cleaned.append([src, dst, val])
    return pd.DataFrame(cleaned, columns=["Источник потока", "Потребитель", "Поток, кВт"])


def process_pdf_cap(file: Any) -> pd.DataFrame:
    """
    Парсит PDF-файл с таблицей лимитов с защитой от шума.
    """
    all_rows = []
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if table: all_rows.extend(table)
    except Exception as e:
        st.error(f"Ошибка чтения PDF: {e}")
        return pd.DataFrame(columns=["начало", "окончание", "Допустимая мощность"])

    cleaned = []
    for row in all_rows:
        if not row or len(row) < 4: continue
        u = normalize_node_strictly(row[1])
        v = normalize_node_strictly(row[2])
        if "начало" in u or "окончание" in v or not u or not v: continue
        if row[0] == "1" and row[1] == "2" and row[2] == "3": continue
        # Пропускная способность может быть пустой (NaN), это валидно (infinite)
        cap = parse_float_safely(row[3], default=None) 
        cleaned.append([u, v, cap])
    return pd.DataFrame(cleaned, columns=["начало", "окончание", "Допустимая мощность"])


def apply_new_data(df_r: pd.DataFrame, df_c: pd.DataFrame) -> None:
    """Обновляет данные в сессии и сбрасывает состояние редакторов."""
    st.session_state.df_req = df_r
    st.session_state.df_cap = df_c
    if "editor_req" in st.session_state: del st.session_state["editor_req"]
    if "editor_cap" in st.session_state: del st.session_state["editor_cap"]
    reset_results()


# Инициализация состояния сессии
if 'df_req' not in st.session_state or 'df_cap' not in st.session_state:
    r, c = generate_synthetic_network(15, 100.0, 0.4)
    st.session_state.df_req = r
    st.session_state.df_cap = c

if 'res' not in st.session_state: st.session_state.res = None
if 'algo_run' not in st.session_state: st.session_state.algo_run = ""
if 'reversed_logical_edges' not in st.session_state: st.session_state.reversed_logical_edges = []
if 'all_logical_edges' not in st.session_state: st.session_state.all_logical_edges = get_logical_edges()
if 'visible_toggles' not in st.session_state: st.session_state.visible_toggles = [('VI.', 'V.')]

# Интерфейс приложения
st.title("⚡ MVP: Оптимизация распределенной электрической сети «Альфа»")
st.markdown("**Интеллектуальная система диспетчеризации (ИИ)** на базе гибридных алгоритмов.")

# Боковая панель: Ввод и настройки
st.sidebar.header("📥 Входные данные")
file_req = st.sidebar.file_uploader("Загрузить Заявки (CSV/Excel/PDF)", type=['csv', 'xlsx', 'pdf'])
file_cap = st.sidebar.file_uploader("Загрузить Ограничения (CSV/Excel/PDF)", type=['csv', 'xlsx', 'pdf'])

if st.sidebar.button("📁 Применить загруженные файлы"):
    try:
        new_req, new_cap = st.session_state.df_req, st.session_state.df_cap
        if file_req:
            if file_req.name.endswith('.pdf'): new_req = process_pdf_req(file_req)
            elif file_req.name.endswith('.xlsx'): new_req = pd.read_excel(file_req)
            else: new_req = pd.read_csv(file_req)
        if file_cap:
            if file_cap.name.endswith('.pdf'): new_cap = process_pdf_cap(file_cap)
            elif file_cap.name.endswith('.xlsx'): new_cap = pd.read_excel(file_cap)
            else: new_cap = pd.read_csv(file_cap)
        apply_new_data(new_req, new_cap)
        st.sidebar.success("Файлы загружены и очищены!")
    except Exception as e:
        st.sidebar.error(f"Критическая ошибка при обработке файлов: {e}")

st.sidebar.markdown("---")
st.sidebar.header("🧪 Сценарный анализ")
with st.sidebar.expander("Параметры генератора"):
    n_req = st.slider("Количество заявок", 1, 50, 15)
    load_val = st.slider("Средняя нагрузка (кВт)", 10, 500, 100)
    bottleneck = st.slider("Степень дефицита (0-1)", 0.0, 1.0, 0.4)
    if st.button("🏗️ Сгенерировать сценарий"):
        df_r, df_c = generate_synthetic_network(n_req, load_val, bottleneck)
        apply_new_data(df_r, df_c)
        st.sidebar.success("Сценарий готов!")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Реверс участков")
edge_options = [f"{u} → {v}" for u, v in st.session_state.all_logical_edges]
selected_edge_str = st.sidebar.selectbox("Добавить на пульт:", ["-- выбор --"] + edge_options)
if st.sidebar.button("➕ Добавить"):
    if selected_edge_str != "-- выбор --":
        u, v = selected_edge_str.split(" → ")
        edge = (u.strip(), v.strip())
        if edge not in st.session_state.visible_toggles:
            st.session_state.visible_toggles.append(edge)
            st.rerun()

for edge in st.session_state.visible_toggles:
    u, v = edge
    active = edge in st.session_state.reversed_logical_edges
    label = f"🔄 {v} → {u}" if active else f"➡️ {u} → {v}"
    c_btn, c_del = st.sidebar.columns([5, 1])
    with c_btn:
        if st.button(label, key=f"rev_{u}_{v}", use_container_width=True):
            if active: st.session_state.reversed_logical_edges.remove(edge)
            else: st.session_state.reversed_logical_edges.append(edge)
            reset_results(); st.rerun()
    with c_del:
        if st.button("✖", key=f"del_{u}_{v}"):
            st.session_state.visible_toggles.remove(edge)
            if active: st.session_state.reversed_logical_edges.remove(edge)
            reset_results(); st.rerun()

st.sidebar.markdown("---")
algo = st.sidebar.radio("🤖 Алгоритм оптимизации", ["Physics-Informed GNN", "Ant Colony (ACO)", "Reinforcement Learning (PPO)"])
with st.sidebar.expander("Параметры обучения"):
    epochs = st.number_input("Итерации/Эпохи", 10, 1000, 30 if algo == "Ant Colony (ACO)" else 150)
    time_limit = st.slider("Лимит времени (мин)", 1, 60, 5)
    early_stop = st.checkbox("Ранняя остановка", value=True)

run_btn = st.sidebar.button("🚀 ЗАПУСТИТЬ РАСЧЕТ", type="primary", use_container_width=True)

# Основная рабочая область
if st.session_state.df_req is not None and st.session_state.df_cap is not None:
    st.subheader("📝 Редактор параметров сети")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("**Таблица 1.1: Заявки на мощность**")
        df_req_edited = st.data_editor(st.session_state.df_req, num_rows="dynamic", width="stretch", on_change=reset_results, key="editor_req")
    with col_e2:
        st.markdown("**Таблица 1.2: Ограничения пропускной способности**")
        df_cap_edited = st.data_editor(st.session_state.df_cap, num_rows="dynamic", width="stretch", on_change=reset_results, key="editor_cap")

    # Валидация топологии в реальном времени с защитой от ошибок парсинга в редакторе
    try:
        current_topo = get_current_topology(st.session_state.reversed_logical_edges)
        nodes, dests, adj, caps, reqs, invalid_reqs = load_network_data(df_req_edited, df_cap_edited, current_topo)
        
        st.markdown("---")
        if invalid_reqs:
            st.error(f"⚠️ **Обнаружено {len(invalid_reqs)} невыполнимых заявок.** Путь физически заблокирован.")
            with st.expander("Детали недостижимых узлов"): st.dataframe(pd.DataFrame(invalid_reqs), width="stretch")
        elif reqs:
            st.success("✅ **Топология корректна.** Все потребители достижимы.")
    except Exception as e:
        st.error(f"Ошибка валидации данных: {e}. Проверьте корректность заполнения таблиц.")
        reqs = {}

    # Выполнение расчета
    if run_btn and reqs:
        with st.spinner(f'Выполняется {algo}...'):
            try:
                if algo == "Ant Colony (ACO)":
                    res = run_aco(nodes, dests, adj, caps, reqs, n_iterations=epochs, time_limit_min=time_limit, early_stop=early_stop)
                elif algo == "Physics-Informed GNN":
                    res = run_gnn(nodes, caps, reqs, epochs=epochs, time_limit_min=time_limit, early_stop=early_stop)
                else:
                    res = run_rl(nodes, dests, adj, caps, reqs, epochs=epochs, time_limit_min=time_limit, early_stop=early_stop)
                st.session_state.res, st.session_state.algo_run = res, algo
                st.success("Расчет завершен успешно!")
            except Exception as e:
                st.error(f"Ошибка при выполнении алгоритма: {e}")

    # Визуализация и анализ результатов
    if st.session_state.res:
        res = st.session_state.res
        final_load = res.get('load_distribution', {})
        total_delivered = sum(res.get('delivered', {}).values())
        total_requested = sum(reqs.values())

        m1, m2 = st.columns(2)
        m1.metric("Общий спрос", f"{total_requested:.2f} кВт")
        m2.metric("Доставленная мощность", f"{total_delivered:.2f} кВт", delta=f"{total_delivered-total_requested:.2f}")

        st.subheader("🛠️ Инспектор потоков")
        edge_list = []
        for (u, v), cap in caps.items():
            load = final_load.get((u, v), 0.0)
            if load > 0.1:
                edge_list.append({'edge': (u, v), 'label': f"{u} → {v} (Загрузка: {load/cap:.1%}, {load:.2f}/{cap:.2f} кВт)", 'ratio': load/cap})
        
        edge_list = sorted(edge_list, key=lambda x: x['ratio'], reverse=True)
        sel_edge_label = st.selectbox("Анализ вклада в нагрузку участка:", ["-- выберите участок --"] + [e['label'] for e in edge_list])

        available_reqs = res['request_flows'].keys()
        req_to_opt = {rk: f"{rk[0]} → {rk[1]} ({res['delivered'].get(rk,0):.2f}/{reqs.get(rk,0):.2f} кВт)" for rk in available_reqs}
        opt_to_key = {v: k for k, v in req_to_opt.items()}

        if sel_edge_label != "-- выберите участок --":
            edge_data = next(e for e in edge_list if e['label'] == sel_edge_label)
            target_edge = edge_data['edge']
            breakdown = [{"key": rk, "label": req_to_opt[rk], "flow": f.get(target_edge, 0)} for rk, f in res['request_flows'].items() if f.get(target_edge, 0) > 0.001]
            with st.expander(f"Детализация участка {target_edge[0]} → {target_edge[1]}", expanded=True):
                cur_sel = st.session_state.get("req_selector", [])
                for b in breakdown:
                    st.checkbox(f"{b['label']} | Вклад: **{b['flow']:.2f} кВт**", value=b['label'] in cur_sel, key=f"chk_{b['key']}", on_change=toggle_request, args=(b['label'],))

        st.subheader("🗺️ Геопространственная визуализация")
        selected_opts = st.multiselect("🔍 Фильтр по заявкам (подсветка путей):", options=list(req_to_opt.values()), key="req_selector")
        selected_load = None
        if selected_opts:
            selected_load = {}
            for opt in selected_opts:
                if opt in opt_to_key:
                    rk = opt_to_key[opt]
                    for edge, val in res['request_flows'].get(rk, {}).items():
                        selected_load[edge] = selected_load.get(edge, 0.0) + val

        fig = draw_interactive_heatmap(nodes, adj, caps, final_load, selected_load)
        st.plotly_chart(fig, width="stretch")

        st.subheader("📊 Метрики качества распределения")
        stats = [{"Источник": s, "Потребитель": d, "Заявка": r, "Факт": res['delivered'].get((s,d),0), "%": (res['delivered'].get((s,d),0)/r*100) if r>0 else 100} for (s,d), r in reqs.items()]
        df_stats = pd.DataFrame(stats)
        c_t1, c_t2 = st.columns([2, 1])
        c_t1.dataframe(df_stats, width="stretch")
        c_t2.plotly_chart(px.histogram(df_stats, x="%", nbins=10, title="Распределение удовлетворенности", color_discrete_sequence=['#4B8BBE']), width="stretch")

    else:
        st.subheader("🗺️ Топологическая схема сети")
        st.info("Выполните расчет для визуализации потоков.")
        try:
            fig = draw_interactive_heatmap(nodes, adj, caps, {})
            st.plotly_chart(fig, width="stretch")
        except: st.warning("Недостаточно данных для отрисовки схемы.")
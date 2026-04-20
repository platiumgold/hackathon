import streamlit as st
import pandas as pd
import graphviz
import pdfplumber
import io

from core.data_loader import load_network_data
from core.aco import run_aco
from core.gnn import run_gnn
from utils.visualization import draw_heatmap
from core.rl import run_rl

st.set_page_config(page_title="MVP Маршрутизации Энергии", layout="wide")

# --- Вспомогательные функции для парсинга PDF ---
def parse_pdf_req(file):
    cleaned = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                for row in table:
                    if not row or len(row) < 4: continue
                    src, dst = str(row[1]).strip(), str(row[2]).strip()
                    if "Источник" in src or "Потребитель" in dst or not src or not dst: continue
                    if dst.lower() in ["итого", "всего"] or "всего" in src.lower(): continue
                    try:
                        val = float(str(row[3]).replace(' ', '').replace(',', '.'))
                        cleaned.append({"Источник потока": src, "Потребитель": dst, "Поток, кВт": val})
                    except ValueError: continue
    return pd.DataFrame(cleaned)

def parse_pdf_cap(file):
    cleaned = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                for row in table:
                    if not row or len(row) < 4: continue
                    u, v = str(row[1]).strip(), str(row[2]).strip()
                    if "начало" in u or "окончание" in v or not u or not v: continue
                    if row[0] == "1" and row[1] == "2" and row[2] == "3": continue
                    try:
                        cap = float(str(row[3]).replace(' ', '').replace(',', '.'))
                        if cap > 0:
                            cleaned.append({"начало": u, "окончание": v, "Допустимая мощность": cap})
                    except ValueError: continue
    return pd.DataFrame(cleaned)

def load_file_to_df(file, file_type="req"):
    if file.name.endswith('.pdf'):
        return parse_pdf_req(file) if file_type == "req" else parse_pdf_cap(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        return pd.read_csv(file)

# --- Инициализация состояния сессии ---
if 'df_req' not in st.session_state:
    st.session_state.df_req = None
if 'df_cap' not in st.session_state:
    st.session_state.df_cap = None
if 'data_confirmed' not in st.session_state:
    st.session_state.data_confirmed = False

def reset_confirmation():
    st.session_state.data_confirmed = False

def generate_graphviz(df_cap):
    dot = graphviz.Digraph(engine='neato')
    dot.attr(overlap='false', splines='true')
    dot.attr('node', shape='circle', style='filled', color='lightblue', fontname='Helvetica')
    for _, row in df_cap.iterrows():
        u, v, cap = str(row.get('начало', '')), str(row.get('окончание', '')), str(row.get('Допустимая мощность', ''))
        if u and v and u != 'nan' and v != 'nan':
            dot.edge(u, v, label=cap, fontsize='10', fontcolor='red')
    return dot

# --- Интерфейс ---
st.title("⚡ MVP: Оптимизация распределенной электрической сети «Альфа»")

# --- БОКОВАЯ ПАНЕЛЬ: Загрузка и Настройки ---
st.sidebar.header("📥 Загрузка данных")
st.sidebar.info("📄 **Поддерживаются форматы: CSV, Excel и PDF!**\n\nПри загрузке PDF (например, сканов ТЗ) система автоматически распознает таблицы. Вы сможете проверить и отредактировать их перед запуском.")

file_req = st.sidebar.file_uploader("Таблица 1.1 (Заявки)", type=['csv', 'xlsx', 'pdf'], on_change=reset_confirmation)
file_cap = st.sidebar.file_uploader("Таблица 1.2 (Сеть)", type=['csv', 'xlsx', 'pdf'], on_change=reset_confirmation)

st.sidebar.markdown("---")
st.sidebar.header("🤖 Выбор и настройка ИИ")
algo = st.sidebar.radio("Алгоритм:", ["Physics-Informed GNN", "Ant Colony (ACO)", "Reinforcement Learning (PPO)"], on_change=reset_confirmation)

model_params = {}
if algo == "Ant Colony (ACO)":
    model_params['n_iterations'] = st.sidebar.slider("Количество итераций", 10, 200, 30)
    model_params['quantum'] = st.sidebar.number_input("Квант потока (кВт)", min_value=1.0, value=10.0)
elif algo == "Physics-Informed GNN":
    model_params['epochs'] = st.sidebar.slider("Эпохи обучения", 100, 1000, 300)
    model_params['cap_penalty'] = st.sidebar.slider("Штраф за перегруз", 0.1, 10.0, 1.0)
elif algo == "Reinforcement Learning (PPO)":
    model_params['epochs'] = st.sidebar.slider("Эпохи PPO", 50, 500, 100)
    model_params['K_paths'] = st.sidebar.slider("Кол-во путей (K)", 2, 10, 5)

# --- ОСНОВНАЯ ОБЛАСТЬ: Обработка данных ---
if file_req and file_cap:
    if st.session_state.df_req is None or st.session_state.df_cap is None:
        with st.spinner("Распознавание файлов..."):
            st.session_state.df_req = load_file_to_df(file_req, "req")
            st.session_state.df_cap = load_file_to_df(file_cap, "cap")

    st.subheader("🕸️ Топология сети")

    # Отрисовка схемы сети
    graph = generate_graphviz(st.session_state.df_cap)
    st.graphviz_chart(graph, use_container_width=True)

    # Кнопки для редактирования
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("✏️ Изменить схему сети (Таблица 1.2)"):
            st.info("Добавляйте, удаляйте или изменяйте связи и мощности. Граф обновится автоматически.")
            edited_cap = st.data_editor(st.session_state.df_cap, num_rows="dynamic", key="editor_cap",
                                        on_change=reset_confirmation)
            # Обновляем состояние
            st.session_state.df_cap = edited_cap

    with col2:
        with st.expander("📋 Посмотреть / Редактировать заявки (Таблица 1.1)"):
            st.info("Отредактируйте источники, потребителей и запрашиваемые мощности.")
            edited_req = st.data_editor(st.session_state.df_req, num_rows="dynamic", key="editor_req",
                                        on_change=reset_confirmation)
            # Обновляем состояние
            st.session_state.df_req = edited_req

    st.markdown("---")

    # Кнопка подтверждения данных
    if not st.session_state.data_confirmed:
        if st.button("✅ Подтвердить входные данные", type="primary"):
            st.session_state.data_confirmed = True
            st.rerun()
    else:
        st.success("Данные подтверждены! Готово к расчету.")
        if st.button("🔄 Изменить данные"):
            st.session_state.data_confirmed = False
            st.rerun()

    # Запуск расчета (показывается только после подтверждения)
    if st.session_state.data_confirmed:
        st.markdown("### Запуск оптимизации")
        if st.button(f"🚀 Запустить расчет: {algo}", type="primary", use_container_width=True):

            with st.spinner('Анализ топологии и парсинг данных...'):
                # Передаем ОТРЕДАКТИРОВАННЫЕ данные из session_state
                nodes, dests, adj, caps, reqs = load_network_data(st.session_state.df_req, st.session_state.df_cap)

            final_load = {}
            total_delivered = 0.0
            total_requested = sum(reqs.values())

            with st.spinner(f'Обучение модели {algo}...'):
                if algo == "Ant Colony (ACO)":
                    res = run_aco(nodes, dests, adj, caps, reqs,
                                  n_iterations=model_params['n_iterations'],
                                  quantum=model_params['quantum'])
                    final_load = res['load_distribution']
                    total_delivered = sum(res['delivered'].values())

                elif algo == "Physics-Informed GNN":
                    flows, node_idx, req_list = run_gnn(nodes, caps, reqs,
                                                        epochs=model_params['epochs'],
                                                        cap_penalty_weight=model_params['cap_penalty'])

                    for req_i, ((src, dst), demand) in enumerate(req_list):
                        in_f = flows[req_i].sum(dim=0)
                        out_f = flows[req_i].sum(dim=1)
                        total_delivered += max(0, (in_f[node_idx[dst]] - out_f[node_idx[dst]]).item())

                    total_edges = flows.sum(dim=0)
                    for u in range(len(nodes)):
                        for v in range(len(nodes)):
                            if total_edges[u, v].item() >= 1.0:
                                final_load[(nodes[u], nodes[v])] = total_edges[u, v].item()

                elif algo == "Reinforcement Learning (PPO)":
                    res = run_rl(nodes, dests, adj, caps, reqs,
                                 epochs=model_params['epochs'],
                                 K_paths=model_params['K_paths'])
                    final_load = res['load_distribution']
                    total_delivered = sum(res['delivered'].values())

            # Вывод метрик
            col1, col2, col3 = st.columns(3)
            col1.metric("Запрошено мощности", f"{total_requested:.3f} кВт")
            col2.metric("Фактически доставлено", f"{total_delivered:.3f} кВт")

            freq_dev = 0.000 if total_delivered >= total_requested else (total_requested - total_delivered) * 0.01
            col3.metric("Частота сети (Гц)", f"{50.000 - freq_dev:.3f} Гц",
                        f"{-freq_dev:.3f} Гц" if freq_dev > 0 else "0.000 Гц")

            # Визуализация
            st.subheader("📊 Распределение потоков (Тепловая карта)")
            fig = draw_heatmap(nodes, adj, caps, final_load)
            st.pyplot(fig)

else:
    st.info("👈 Пожалуйста, загрузите оба файла с данными (Таблицы 1.1 и 1.2) через боковую панель.")
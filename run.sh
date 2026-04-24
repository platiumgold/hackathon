#!/bin/bash
echo "======================================================"
echo "  Запуск MVP: Оптимизация электрической сети Альфа"
echo "  (Linux / WSL Offline Mode)"
echo "======================================================"

# Проверка наличия streamlit
if ! command -v streamlit &> /dev/null
then
    echo "[ОШИБКА] Streamlit не найден. Пожалуйста, установите его: pip install streamlit"
    exit
fi

# Запуск с флагами отключения телеметрии
streamlit run app.py --browser.gatherUsageStats=false --server.headless=false

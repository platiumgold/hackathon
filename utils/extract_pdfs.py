import sys
import subprocess
import io
import pandas as pd
import csv
from typing import List, Any, Optional
try:
    import pdfplumber
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfplumber"])
    import pdfplumber

# Настройка вывода в UTF-8 для корректного отображения кириллицы
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def clean_value(val: Any) -> float:
    """
    Очищает строковое значение от пробелов и корректирует десятичный разделитель.
    
    Args:
        val: Значение из ячейки таблицы.
    """
    if not val: return 0.0
    try:
        return float(str(val).replace(' ', '').replace(',', '.'))
    except ValueError:
        return 0.0


def process_table_1_1(raw_rows: List[List[Any]]) -> None:
    """
    Обрабатывает сырые данные Таблицы 1.1 (Объемы производства) и сохраняет в CSV.
    
    Args:
        raw_rows: Список строк из PDF-таблицы.
    """
    cleaned = []
    for row in raw_rows:
        if not row or len(row) < 4: continue
        src = str(row[1]).strip() if row[1] else ""
        dst = str(row[2]).strip() if row[2] else ""
        
        # Фильтрация заголовков и пустых строк
        if "Источник" in src or "Потребитель" in dst or not src or not dst:
            continue
        # Фильтрация итоговых строк
        if dst.lower() == "итого" or "всего" in src.lower():
            continue
            
        val = clean_value(row[3])
        cleaned.append([src, dst, val])
    
    df = pd.DataFrame(cleaned, columns=["Источник потока", "Потребитель", "Поток, кВт"])
    df.to_csv("real_task_table_1_1.csv", index=False, encoding='utf-8')
    print("Результат: Таблица 1.1 сохранена в real_task_table_1_1.csv")


def process_table_1_2(raw_rows: List[List[Any]]) -> None:
    """
    Обрабатывает сырые данные Таблицы 1.2 (Допустимая мощность) и сохраняет в CSV.
    
    Args:
        raw_rows: Список строк из PDF-таблицы.
    """
    cleaned = []
    for row in raw_rows:
        if not row or len(row) < 4: continue
        
        u = str(row[1]).strip() if row[1] else ""
        v = str(row[2]).strip() if row[2] else ""
        
        # Фильтрация заголовков и технических строк (1, 2, 3...)
        if "начало" in u or "окончание" in v or not u or not v:
            continue
        if row[0] == "1" and row[1] == "2" and row[2] == "3":
            continue
            
        cap = clean_value(row[3])
        if cap > 0:
            cleaned.append([u, v, cap])
            
    df = pd.DataFrame(cleaned, columns=["начало", "окончание", "Допустимая мощность"])
    df.to_csv("real_task_table_1_2.csv", index=False, encoding='utf-8')
    print("Результат: Таблица 1.2 сохранена в real_task_table_1_2.csv")


def extract_and_clean() -> None:
    """Главная функция для пакетной обработки PDF-файлов задания."""
    # Обработка Таблицы 1.1
    try:
        with pdfplumber.open("1.1._Табл. 1.1_Объемы производства эл.эн..pdf") as pdf:
            all_rows = []
            for page in pdf.pages:
                table = page.extract_table()
                if table: all_rows.extend(table)
            process_table_1_1(all_rows)
    except FileNotFoundError:
        print("Файл Таблицы 1.1 не найден.")

    # Обработка Таблицы 1.2
    try:
        with pdfplumber.open("1.2._Табл. 1.2_ Допустимая мощность потоков.pdf") as pdf:
            all_rows = []
            for page in pdf.pages:
                table = page.extract_table()
                if table: all_rows.extend(table)
            process_table_1_2(all_rows)
    except FileNotFoundError:
        print("Файл Таблицы 1.2 не найден.")


if __name__ == "__main__":
    extract_and_clean()

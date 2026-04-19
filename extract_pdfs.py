import sys
import subprocess
import io

# Set console output to UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

try:
    import pdfplumber
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfplumber"])
    import pdfplumber
import pandas as pd
import csv

def clean_value(val):
    if not val: return 0.0
    try:
        # Remove spaces and replace decimal comma with dot
        return float(str(val).replace(' ', '').replace(',', '.'))
    except ValueError:
        return 0.0

def process_table_1_1(raw_rows):
    # Header: Номер строки,Источник потока,Потребитель,"Поток, кВт"
    cleaned = []
    for row in raw_rows:
        if not row or len(row) < 4: continue
        src = str(row[1]).strip() if row[1] else ""
        dst = str(row[2]).strip() if row[2] else ""
        
        # Skip headers and empty rows
        if "Источник" in src or "Потребитель" in dst or not src or not dst:
            continue
        # Skip totals
        if dst.lower() == "итого" or "всего" in src.lower():
            continue
            
        val = clean_value(row[3])
        cleaned.append([src, dst, val])
    
    df = pd.DataFrame(cleaned, columns=["Источник потока", "Потребитель", "Поток, кВт"])
    df.to_csv("real_task_table_1_1.csv", index=False, encoding='utf-8')
    print("Saved cleaned table_1_1.csv")

def process_table_1_2(raw_rows):
    cleaned = []
    for row in raw_rows:
        if not row or len(row) < 4: continue
        
        u = str(row[1]).strip() if row[1] else ""
        v = str(row[2]).strip() if row[2] else ""
        
        # Skip headers and trash index rows (1,2,3,4)
        if "начало" in u or "окончание" in v or not u or not v:
            continue
        if row[0] == "1" and row[1] == "2" and row[2] == "3":
            continue
            
        cap = clean_value(row[3])
        if cap > 0:
            cleaned.append([u, v, cap])
            
    df = pd.DataFrame(cleaned, columns=["начало", "окончание", "Допустимая мощность"])
    df.to_csv("real_task_table_1_2.csv", index=False, encoding='utf-8')
    print("Saved cleaned table_1_2.csv")

def extract_and_clean():
    # Table 1.1
    with pdfplumber.open("1.1._Табл. 1.1_Объемы производства эл.эн..pdf") as pdf:
        all_rows = []
        for page in pdf.pages:
            table = page.extract_table()
            if table: all_rows.extend(table)
        process_table_1_1(all_rows)

    # Table 1.2
    with pdfplumber.open("1.2._Табл. 1.2_ Допустимая мощность потоков.pdf") as pdf:
        all_rows = []
        for page in pdf.pages:
            table = page.extract_table()
            if table: all_rows.extend(table)
        process_table_1_2(all_rows)

if __name__ == "__main__":
    extract_and_clean()

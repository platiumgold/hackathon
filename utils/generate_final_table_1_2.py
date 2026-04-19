import pandas as pd

# 1. Existing data from Table 1.2 (keeping dots as they are in the PDF)
existing_edges = {
    ('P', 'XI.'): 2670.3,
    ('XVIII.', '6'): 4878.2,
    ('XVIII.', 'XI.'): 3644.6,
    ('X.', 'IX.'): 12900.9,
    ('IX.', 'XV.'): 8991.7,
    ('C', 'X.'): 4503.4,
    ('M', 'X.'): 2612.5,
    ('IX.', 'I.'): 6269.8,
    ('XI.', 'X.'): 5901.7,
    ('I.', 'III.'): 11578.7,
    ('II.', 'I.'): 7057.1,
    ('III.', 'XVI.'): 20757.7,
    ('XVII.', 'IV.'): 12672.7,
    ('V.', 'XVII.'): 7322.4,
    ('VI.', 'V.'): 5034.8,
    ('V.', 'VIII.'): 1614.1,
    ('J', 'VIII.'): 3573.7,
    ('XVI.', '1'): 27945.0,
    ('XII.', 'VI.'): 10516.7,
    ('VI.', '15'): 6207.4,
    ('XV.', 'XIV.'): 11089.0,
    ('XIV.', 'XII.'): 14548.4,
    ('XII.', 'XIII.'): 5087.7,
    ('XIII.', '13'): 1571.2,
    ('XIII.', '24'): 1864.6,
    ('13', '25'): 1381.1,
    ('O', 'VII.'): 1762.5,
    ('N', 'VII.'): 2529.9,
    ('VII.', 'VI.'): 3501.2,
}

# 2. Edges identified from the image diagram (with dots for Roman nodes)
image_edges = [
    ('A', 'XVIII.'), ('B', 'X.'), ('C', 'X.'), ('D', 'XV.'), ('E', 'XIV.'),
    ('F', 'II.'), ('G', 'II.'), ('H', 'III.'), ('I', 'V.'), ('J', 'VIII.'),
    ('K', 'XVI.'), ('L', 'XVI.'), ('M', 'X.'), ('N', 'VII.'), ('O', 'VII.'), ('P', 'XI.'),
    
    ('XVIII.', '6'), ('XVIII.', '7'), ('XVIII.', '5'), ('XVIII.', 'XI.'),
    ('XI.', '9'), ('XI.', '4'), ('XI.', '11'), ('XI.', 'X.'),
    ('X.', '10'), ('X.', '12'), ('X.', 'IX.'),
    ('IX.', '8'), ('IX.', 'XV.'),
    ('XV.', '29'), ('XV.', 'XIV.'),
    ('XIV.', '23'), ('XIV.', '22'), ('XIV.', '28'), ('XIV.', 'XII.'),
    ('XII.', '21'), ('XII.', '20'), ('XII.', '3'), ('XII.', '17'), ('XII.', '19'), ('XII.', 'VI.'), ('XII.', 'XIII.'),
    ('XIII.', '13'), ('XIII.', '18'), ('XIII.', '24'), ('XIII.', '14'), ('XIII.', '25'),
    ('VI.', '27'), ('VI.', '35'), ('VI.', '36'), ('VI.', 'VII.'), ('VI.', 'V.'),
    ('VII.', 'VI.'),
    ('V.', '32'), ('V.', 'VIII.'),
    ('VIII.', '37'), ('VIII.', 'XVII.'),
    ('XVII.', '33'), ('XVII.', '34'), ('XVII.', 'XVI.'),
    ('XVI.', '1'), ('XVI.', '2'), ('XVI.', '38'), ('XVI.', 'IV.'),
    ('IV.', 'III.'), ('III.', 'IV.'), ('III.', '30'), ('III.', '31'), ('III.', 'I.'),
    ('I.', 'II.'), ('II.', 'I.'), ('I.', 'L.'), ('L.', 'IX.')
]

# 3. Merge logic
VIRTUAL_CAP = 40000.0
final_data = []
seen = set()

for (u, v), cap in existing_edges.items():
    final_data.append([u, v, cap])
    seen.add((u, v))

for (u, v) in image_edges:
    if (u, v) not in seen:
        final_data.append([u, v, VIRTUAL_CAP])
        seen.add((u, v))

# 4. Save to CSV
df = pd.DataFrame(final_data, columns=["начало", "окончание", "Допустимая мощность"])
df.to_csv("table_1_2.csv", index=False, encoding='utf-8')
print(f"Final table_1_2.csv created with {len(df)} edges. Roman numerals have dots.")

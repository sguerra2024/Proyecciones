import os
import pandas as pd

files = [
    'Produccion Astroflores a la Semana 26_Seguimiento.xlsx',
    'Produccion Astroflores BL25-26-27-28 a la Semana 27_Seguimiento.xlsx',
    'Produccion Astroflores a la Semana 28_seguimiento.xlsx',
]

for f in files:
    path = os.path.join(os.getcwd(), f)
    print(f'\n=== {f} ===')
    xls = pd.ExcelFile(path)
    print('Sheets:', xls.sheet_names)
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        print(f'Sheet {sheet} shape: {df.shape}')
        print(df.head(10).to_string(index=False))
        print('---')

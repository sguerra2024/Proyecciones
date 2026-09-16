import pandas as pd
from projection_core import (
    prepare_crop_age_series,
    has_sufficient_pattern_history,
    calculate_age_aligned_normalized_stems_mse,
)

path = 'Produccion Astroflores BL25-26-27-28 a la Semana 37_entrenamiento.xlsx'
df = pd.read_excel(path)
groups = list(df.dropna(subset=['Bloque&Varid']).groupby('Bloque&Varid'))
print('SHAPE', df.shape)
print('VARIETIES', len(groups))


def diagnose(target):
    target_rows = df[df['Bloque&Varid'].astype(str).eq(target)]
    target_age = prepare_crop_age_series(target_rows)
    print('\nTARGET', target, 'rows', len(target_rows), 'complete4', len(target_rows.dropna(
        subset=['Anio', 'Semana', 'Tallos/m2', 'Produccion'])), 'age', len(target_age))
    stats = {'same': 0, 'short': 0, 'mse_none': 0,
             'age_empty': 0, 'accepted': 0}
    accepted = []
    for name, group in groups:
        name = str(name)
        if name == target:
            stats['same'] += 1
            continue
        if not has_sufficient_pattern_history(target_rows, group, required_columns=('Anio', 'Semana', 'Tallos/m2', 'Produccion')):
            stats['short'] += 1
            continue
        mse = calculate_age_aligned_normalized_stems_mse(target_rows, group)
        if mse is None:
            stats['mse_none'] += 1
            continue
        age = prepare_crop_age_series(group)
        if age.empty:
            stats['age_empty'] += 1
            continue
        stats['accepted'] += 1
        accepted.append((name, len(group), len(group.dropna(subset=[
                        'Anio', 'Semana', 'Tallos/m2', 'Produccion'])), round(mse, 4), str(age['Etapa_cultivo'].iloc[-1])))
    print('STATS', stats)
    print('ACCEPTED', sorted(accepted, key=lambda x: x[3])[:15])
    return stats


for target in ['36013LUMIA', '36013COUNTRY BLUES', '36013LORRAINE', '36013VENDELA', '36011LUMIA']:
    if target in set(df['Bloque&Varid'].astype(str)):
        diagnose(target)

summary = []
for name, target_rows in groups:
    target = str(name)
    stats = diagnose(target)
    summary.append((target, len(target_rows), len(target_rows.dropna(subset=[
                   'Anio', 'Semana', 'Tallos/m2', 'Produccion'])), stats['accepted'], stats['short'], stats['mse_none'], stats['age_empty']))
print('\nZERO_ACCEPTED')
for row in summary:
    if row[3] == 0:
        print(row)
print('ACCEPTED_DISTRIBUTION', pd.Series(
    [row[3] for row in summary]).value_counts().sort_index().to_dict())

import sys
import pandas as pd

if len(sys.argv) < 2:
    print('Usage: python3 length.py path1')

USECOLS = ['URL', 'label']

phishing_label = int(input('enter phishing label: '))
requested = 20
requested_enum = range(requested)

legitimate_label = 1 - phishing_label
path = sys.argv[1]
df = pd.read_csv(path, usecols = USECOLS)
df_p = df[df[USECOLS[1]] == phishing_label].copy(deep = True)
df_l = df[df[USECOLS[1]] == legitimate_label].copy(deep = True)

col_len = 'URL_LEN'
col_idx = 'IDX'

df_p[col_len] = df_p[USECOLS[0]].apply(lambda elem : len(elem))
df_l[col_len] = df_l[USECOLS[0]].apply(lambda elem : len(elem))

df_p = df_p.sort_values(by = col_len, ascending = False)
df_p = df_p[:requested]
df_p[col_idx] = range(requested)
df_p.to_csv(f'phi_{path}', index = False)

df_l = df_l.sort_values(by = col_len, ascending = False)
df_l = df_l[:requested]
df_l[col_idx] = range(requested)
df_l.to_csv(f'leg_{path}', index = False)

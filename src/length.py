import sys
import pandas as pd

if len(sys.argv) < 2:
    print('Usage: python3 length.py path1')

USECOLS = ['URL', 'label']

phishing_label = int(input('enter phishing label: '))
requested = 1024

legitimate_label = 1 - phishing_label
path = sys.argv[1]
df = pd.read_csv(path, usecols = USECOLS)
df_p = df[df[USECOLS[1]] == phishing_label].copy(deep = True)
df_l = df[df[USECOLS[1]] == legitimate_label].copy(deep = True)

new_col = 'URL_LEN'

df_p[new_col] = df_p[USECOLS[0]].apply(lambda elem : len(elem))
df_l[new_col] = df_l[USECOLS[0]].apply(lambda elem : len(elem))

df_p = df_p.sort_values(by = new_col, ascending = False)
df_p[:requested].to_csv(f'sorted_len_phi_{path}', index = False)

df_l = df_l.sort_values(by = new_col, ascending = False)
df_l[:requested].to_csv(f'sorted_len_leg_{path}', index = False)

import sys
import pandas as pd
import re

cont = 0

def detect_percert_encoding(name: str):
    global cont
    if re.search(r"%[0-9A-Fa-f]{2}", name):
        cont += 1

if len(sys.argv) < 2:
    print('Usage: python3 length.py path1')

USECOLS = ['URL', 'label']

phishing_label = int(input('enter phishing label: '))
legitimate_label = 1 - phishing_label

path = sys.argv[1]
df = pd.read_csv(path, usecols = USECOLS)
df_p = df[df[USECOLS[1]] == phishing_label].copy(deep = True)
df_l = df[df[USECOLS[1]] == legitimate_label].copy(deep = True)

total_phi = df_p.shape[0]
df_p[USECOLS[0]].apply(lambda elem : detect_percert_encoding(elem))
print(f'phishing: total[{total_phi}] p_end[{cont}]')

total_leg = df_l.shape[0]
cont = 0
df_l[USECOLS[0]].apply(lambda elem : detect_percert_encoding(elem))
print(f'legitimate: total[{total_leg}] p_end[{cont}]')



import sys
from selection.flex_selector import FeatureSelector
from extraction.flex_extractor import extract
from utils.wrappers import export
import pandas as pd

if len(sys.argv) <= 2:
    print("Usage: python3 extract.py path1 path2")
    exit(1)

USECOLS = ['URL', 'label']
DTYPES = {'URL': 'string', 'label': 'int8'}

size = 4
request = 1024
l_phishing = int(input("label if phishing: "))
l_legitimate = 1 - l_phishing
name_sel = input("selection file name: ")

df = pd.read_csv(sys.argv[1], usecols = USECOLS, dtype = DTYPES)
se = FeatureSelector(size, request)
print("selecting 4-grams...")
se.select(df, l_phishing, l_legitimate)
print("exporting...")
se.dump_info(name_sel)

name_ext = input("extraction file name: ")
print("extracting")
ex = extract(f'{name_sel}', [sys.argv[2]])
print("exporting")
export(ex, name_ext)

import sys
from extraction.extractor import FeatureSelector
from extraction.extractor import strip_url
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python3 extract.py path1")
    exit(1)

USECOLS = ['URL', 'label']
DTYPES = {'URL': 'string', 'label': 'int8'}

size = 4
request = 4096
l_phishing = int(input("label if phishing: "))
l_legitimate = 1 - l_phishing
source = input("source: ")
force  = input("force selection based only on frequency (y/n): ")
force_freq_sel = force == 'y'
df = pd.read_csv(sys.argv[1], usecols = USECOLS, dtype = DTYPES)
se = FeatureSelector(size, request)
print("selecting 4-grams...")
se.select(df, l_phishing, l_legitimate, force_freq_sel)
print("exporting...")
se.dump_info(source)

# df['URL'] = df['URL'].apply(strip_url)

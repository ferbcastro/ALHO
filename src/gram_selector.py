import sys
from extraction.extractor import FeatureSelector
from extraction.extractor import strip_url
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python3 extract.py path1 path2 ...")
    exit(1)

USECOLS = ['URL', 'label']
DTYPES = {'URL': 'string', 'label': 'int8'}

size = 4
request = 2046
l_phishing = int(input("Label if phishing: "))
l_legitimate = 1 - l_phishing
for path in sys.argv[1:]:
    source = input("Source: ")
    df = pd.read_csv(path, usecols = USECOLS, dtype = DTYPES)
    se4 = FeatureSelector(size, request)
    print("selecting 4-grams...")
    se4.select(df, l_phishing, l_legitimate)
    print("exporting...")
    se4.dump_info(source)

# df['URL'] = df['URL'].apply(strip_url)

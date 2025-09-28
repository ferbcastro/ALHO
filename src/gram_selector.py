import sys
from extraction.extractor import FeatureSelector
from extraction.extractor import strip_url
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python3 extract.py path1 path2 ...")
    exit(1)

USECOLS = ['URL', 'label']
DTYPES = {'URL': 'string', 'label': 'int8'}

frames = []
for path in sys.argv[1:]:
    tmp = pd.read_csv(path, usecols = USECOLS, dtype = DTYPES)
    frames.append(tmp)
df = pd.concat(frames, ignore_index = True)
# df['URL'] = df['URL'].apply(strip_url)

# size = 3
# request = 1024
# se3 = FeatureSelector(size, request)
# print("selecting 3-grams...")
# se3.select(df)
# print("exporting...")
# se3.dump_info()

size = 4
request = 2046
se4 = FeatureSelector(size, request)
print("selecting 4-grams...")
se4.select(df)
print("exporting...")
se4.dump_info()

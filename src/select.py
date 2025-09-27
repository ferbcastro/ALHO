import sys
from extraction.extractor import FeatureSelector
from extraction.extractor import strip_url
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python3 extract.py path1 path2 ...")
    exit(1)

frames = []
for path in sys.argv[1:]:
    tmp = pd.read_csv(path)
    frames.append(tmp)
df = pd.concat(frames, ignore_index = True)
df['URL'] = df['URL'].apply(strip_url)

size = 3
request = 1024
threshold = 0.3

se = FeatureSelector(size, request, threshold)
print("selecting...")
se.select(df)
print("exporting...")
se.dump_info()

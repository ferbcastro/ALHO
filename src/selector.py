import extraction.FeatureSelector as fs
from extraction.extractor import split
from multiprocessing import Pool, cpu_count
import sys
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python3 extract.py path1 path2 ...")
    exit(1)

frames = []
for path in sys.argv[1:]:
    tmp = pd.read_csv(path)
    frames.append(tmp)
chunks, num_chunks = split(frames)

size = 3
request = 1000
threshold = 0.3

se = fs.FeatureSelector(size, request, threshold)
print("selecting...")
se.select(chunks, num_chunks)
print("exporting...")
se.dump_info()

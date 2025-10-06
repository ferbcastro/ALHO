from extraction.bigram_extractor import extract
from utils.wrappers import export
import sys

if len(sys.argv) < 2:
    print("Usage: python3 extract.py path1 path2 ...")
    exit(1)

name = input("Name: ")
print("extracting...")
res = extract(sys.argv[1:])
print("exporting...")
export(res, name)

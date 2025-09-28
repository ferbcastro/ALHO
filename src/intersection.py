import os
import sys
import pandas as pd

if len(sys.argv) < 3:
    print("Usage: python3 extract.py path1 path2")
    exit(1)

field = input("Base column: ")
name = input("New csv name: ")
df1 = pd.read_csv(sys.argv[1])
df2 = pd.read_csv(sys.argv[2])
df = pd.merge(df1, df2, on = field, how = 'inner', suffixes = ("_1", "_2"))
df.to_csv(name, index = False)

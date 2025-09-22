import extraction.extractor as fe
import sys

if len(sys.argv) < 2:
    print("Usage: python3 extract.py path1 paht2 ...")
    exit(1)

ex = fe.FeatureExtractor(sys.argv[1:])
print("extracting...")
ex.extract()
print("exporting...")
ex.export("out.csv")

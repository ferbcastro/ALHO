import pandas as pd
from urllib.parse import urlparse
from typing import *
import re
import string
import collections
import heapq

_CHAR_SPACE = string.printable[:-6] # printable characters except whitespaces
_CHAR_SPACE_LEN = len(_CHAR_SPACE)
_CHAR_INDEX = {c: i for i, c in enumerate(_CHAR_SPACE)}

# Helper functions
# Strip scheme and characters outside _CHAR_SPACE
def strip_url(url: str) -> str:
    url = "".join(char for char in url if char in _CHAR_SPACE)

    if (scheme := urlparse(url).scheme):
        url = re.sub(f"^{scheme}://", "", url)

    return url

# distribution_unit = 1/(len(url)-1)
def bigram_dist(url: str) -> Tuple[List[float], List[int]]:

    url_len = len(url)
    total_bigrams = url_len - 1
    bigram_presence = [0] * (_CHAR_SPACE_LEN**2)
    distribution_unit = 1/total_bigrams

    for i in range(total_bigrams):
        idx = _CHAR_INDEX[url[i]] * _CHAR_SPACE_LEN + _CHAR_INDEX[url[i + 1]]
        bigram_presence[idx] = 1

    return bigram_presence

def process_row(row: pd.Series) -> pd.Series:
    url = row["url"]
    url_s = strip_url(url)
    bigram_presence = bigram_dist(url_s)

    features ={"label" : row["label"]}
    for i, j in enumerate(bigram_presence):
        idx1 = i // _CHAR_SPACE_LEN
        idx2 = i % _CHAR_SPACE_LEN
        big = _CHAR_SPACE[idx1] + _CHAR_SPACE[idx2]
        features.update({big:j})

    return pd.Series(features)

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(process_row, axis=1)

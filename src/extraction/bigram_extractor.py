from multiprocessing import Pool, cpu_count
import pandas as pd
import string
import math

import utils.wrappers as wp
import utils.urls as ul

URL_FIELD = 'URL'
LABEL_FIELD = 'label'

CHAR_SPACE = string.printable[:-6] # printable characters except whitespaces
CHAR_SPACE_LEN = len(CHAR_SPACE)
CHAR_INDEX = {c: i for i, c in enumerate(CHAR_SPACE)}

def extract(paths) -> pd.DataFrame:
    df = wp.concat(paths)
    df[URL_FIELD] = df[URL_FIELD].map(ul.strip_url)

    n_procs = cpu_count()
    chunk_size = math.ceil(df.shape[0] / n_procs)
    chunks = []
    for i in range(n_procs):
        start_index = i * chunk_size
        end_index = (i + 1) * chunk_size
        chunk = df.iloc[start_index:end_index]
        chunks.append(chunk)

    with Pool(n_procs) as pool:
        results = pool.map(extract_bigrams, chunks)

    return pd.concat(results, ignore_index = True)

def extract_bigrams(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(process_row, axis = 1)

def process_row(row: pd.Series) -> pd.Series:
    url = row[URL_FIELD]
    features = {
            "url"   : row[URL_FIELD],
            "label" : row[LABEL_FIELD]
            }

    bigram_presence = presence(url)

    for i, j in enumerate(bigram_presence):
        idx1 = i // CHAR_SPACE_LEN
        idx2 = i % CHAR_SPACE_LEN
        big = CHAR_SPACE[idx1] + CHAR_SPACE[idx2]
        features.update({big:j})

    return pd.Series(features)

def presence(url: str) -> list[int]:
    url_len = len(url)
    total_bigrams = url_len - 1
    presence = [0] * (CHAR_SPACE_LEN**2)

    for i in range(total_bigrams):
        idx = CHAR_INDEX[url[i]] * CHAR_SPACE_LEN + CHAR_INDEX[url[i + 1]]
        presence[idx] = 1

    return presence
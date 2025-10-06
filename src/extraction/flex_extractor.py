from multiprocessing import Pool, cpu_count
import pandas as pd

import utils.wrappers as wp
import utils.urls as ul

import math

URL_FIELD = 'URL'
LABEL_FIELD = 'label'
GRAM_NAME_FIELD = 'gram_names'

ngrams = set()
gram_size = 4

def extract(grampath, paths) -> pd.DataFrame:
    df = wp.concat(paths)

    read_features(grampath)

    n_procs = cpu_count()
    chunk_size = math.ceil(df.shape[0] / n_procs)
    chunks = []
    for i in range(n_procs):
        start_index = i * chunk_size
        end_index = (i + 1) * chunk_size
        chunk = df.iloc[start_index:end_index]
        chunks.append(chunk)

    with Pool(n_procs) as pool:
        results = pool.map(flex_extract, chunks)

    return pd.concat(results, ignore_index = True)

def process_row(row: pd.Series) -> pd.Series:
    global ngrams
    global gram_size
    url = row[URL_FIELD]
    features = {
            "url"   : row[URL_FIELD],
            "label" : row[LABEL_FIELD]
            }

    for elem in ngrams:
        features.update({elem:0})
    url_len = len(url)
    if url_len >= gram_size:
        for i in range(url_len - gram_size):
            key = url[i : i + gram_size]
            if key in ngrams:
                features[key] = 1

    return pd.Series(features)

def flex_extract(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(process_row, axis = 1)

def read_features(path):
    global ngrams
    df = pd.read_csv(path, usecols=[GRAM_NAME_FIELD])
    ngrams_list = df[GRAM_NAME_FIELD].to_list()
    ngrams = set(ngrams_list)
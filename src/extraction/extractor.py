from extraction.features import extract_features
from multiprocessing import Pool, cpu_count

import pandas as pd
import math
import sys

class FeatureExtractor:
    extract_all: bool
    num_grams: int # ignored if extract_all is True
    gram_size: int

    char_space = string.printable[:-6] # printable characters except whitespaces
    char_space_len = len(char_space)
    char_index = {c: i for i, c in enumerate(char_space)}

    selected_grams: set # ignored if extract_all is True

    df: pd.DataFrame

    def __init__(self, paths, extract_all, num_grams=-1, gram_size=2) -> None:
        self.extract_all = extract_all
        if extract_all == False
            self.num_grams = num_grams
        self.gram_size = gram_size
        df1 = pd.read_csv(paths[0])
        frames = [df1]
        for path in paths[1:]:
            tmp = pd.read_csv(path)
            frames.append(tmp)

        self.df = pd.concat(frames, ignore_index=True)

    def extract(self) -> None:

        n_procs = cpu_count()
        chunk_size = math.ceil(self.df.shape[0] / n_procs)
        chunks = []

        for i in range(n_procs):
            start_index = i * chunk_size
            end_index = (i + 1) * chunk_size
            df_chunk = self.df.iloc[start_index:end_index]
            chunks.append(df_chunk)

        with Pool(n_procs) as pool:
            results = pool.map(self._extract_features, chunks)

        self.df = pd.concat(results)

    def export(self, path: str) -> None:
        self.df.to_csv(path, index=False)

    # Strip scheme and characters outside CHAR_SPACE
    def _strip_url(self, url: str) -> str:
        url = "".join(char for char in url if char in self.char_space)

        if (scheme := urlparse(url).scheme):
            url = re.sub(f"^{scheme}://", "", url)

        return url

    def _ngram_extract_all(url: str) -> List[int]:
        url_len = len(url)
        total_ngrams = url_len - (gram_size - 1)
        presence = [0] * (self.char_space_len**gram_size)

        for i in range(total_ngrams):
            idx = self.char_index[url[i]] * char_space_len +
                    self.char_index[url[i + 1]]
            presence[idx] = 1

    def _ngram_extract_partial(url: str) -> List[int]:

    def _ngram_extract(url: str) -> List[int]:
        if self.extract_all == True:
            ngram_presence = self._ngram_extract_all(url)
        else
            ngram_presence = self._ngram_extract_partial(url)

        return ngram_presence

    def _process_row(row: pd.Series) -> pd.Series:
        url = row["url"]
        features = {
                "url"   : row["url"],
                "label" : row["label"]
                }

        url_s = strip_url(url)
        bigram_presence = bigram_dist(url_s)

        for i, j in enumerate(bigram_presence):
            idx1 = i // _CHAR_SPACE_LEN
            idx2 = i % _CHAR_SPACE_LEN
            big = _CHAR_SPACE[idx1] + _CHAR_SPACE[idx2]
            features.update({big:j})

        return pd.Series(features)

    def _extract_features(df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(process_row, axis=1)

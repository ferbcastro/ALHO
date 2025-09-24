from multiprocessing import Pool, cpu_count
from urllib.parse import urlparse
from typing import *

import pandas as pd
import math
import sys
import re

CHAR_SPACE = string.printable[:-6] # printable characters except whitespaces
CHAR_SPACE_LEN = len(CHAR_SPACE)
CHAR_INDEX = {c: i for i, c in enumerate(CHAR_SPACE)}

class FeatureExtractor:
    org_df: pd.DataFrame
    mod_df: pd.DataFrame

    def __init__(self, paths) -> None:
        df1 = pd.read_csv(paths[0])
        frames = [df1]
        for path in paths[1:]:
            tmp = pd.read_csv(path)
            frames.append(tmp)

        df = pd.concat(frames, ignore_index=True)
        df['url'] = df['url'].apply(self._strip_url)
        org_df = df

    def extract(self, prep_func, extraction_func) -> None:
        n_procs = cpu_count()
        chunk_size = math.ceil(self.org_df.shape[0] / n_procs)
        chunks = []

        for i in range(n_procs):
            start_index = i * chunk_size
            end_index = (i + 1) * chunk_size
            df_chunk = self.org_df.iloc[start_index:end_index]
            chunks.append(df_chunk)

        if prep_func:
            prep_func(chunks, n_procs) # any preprocessing needed
        with Pool(n_procs) as pool:
            results = pool.map(extraction_func, chunks)
        self.mod_df = pd.concat(results, ignore_index=True)

    def export(self, path: str) -> None:
        self.mod_df.to_csv(path, index=False)

    # Strip scheme and characters outside CHAR_SPACE
    def _strip_url(self, url: str) -> str:
        url = "".join(char for char in url if char in CHAR_SPACE)

        if (scheme := urlparse(url).scheme):
            url = re.sub(f"^{scheme}://", "", url)

        return url

class FeatureSelector:
    num_not_phishing: int
    num_phishing: int
    requested: int
    gram_size: int
    threshold: float
    selected: int
    space: set

    def __init__(self, gram_size, requested_grams, threshold) -> None:
        self.space = {}
        self.selected = self.num_not_phishing = self.num_phishing = 0
        if gram_size < 0:
            self.gram_size = 2
            print(f'gram_size set to [{size}]', self.gram_size)
        else:
            self.gram_size = gram_size

        max_grams = CHAR_SPACE_LEN ** self.gram_size
        if requested_grams > max_grams:
            self.requested = max_grams
            print(f'requested_grams set to {num}', self.requested)
        else:
            self.requested = requested_grams

        if threshold < 0 or threshold > 1:
            self.threshold = 0.5
            print(f'threshold set to {num}', self.threshold)
        else:
            self.threshold = threshold

    def select(self, chunks, num_chunks) -> {}:
        with Pool(num_chunks) as pool:
            results = pool.map(self._build_dictionary, chunks)

        total_grams_dict = dict()
        for d in results:
            self._merge_dict(total_grams_dict, d)

        view = total_grams_dict.items()
        sorted_grams_list = sorted(view, lambda it : it[1][0], reverse=True)
        self._select_features(sorted_grams_list)
        return self.space.copy()

    def statistics(self) -> None:
        print(f'Number of selected grams [{num}]', self.selected)
        print(f'Threshold set to [{num}]', self.threshold)

    def _build_dictionary(self, df: pd.DataFrame) -> dict[str : tuple[int, int]]:
        dct = dict()
        df.apply(lambda row : self._build_dict_process_row(row, dct), axis = 1)
        return dct

    def _build_dict_process_row(self, row: pd.Series, dct) -> None:
        url = row['url']
        label = row['label']
        url_len = len(url)

        if url_len < self.gram_size:
            return

        for i in range(urlSize - gramSize):
            key = url[i : i + gramSize]
            if key in dct:
                dct['key'][0] += 1
            else:
                dct.update({key : [1, 0]})
            if label == 0:
                dct['key'][1] += 1
                self.num_not_phishing += 1
            else:
                self.num_phishing += 1

    # merge in dct1
    def _merge_dict(
        self,
        dct1: dict[str, tuple[int, int]],
        dct2: dict[str, tuple[int, int]]
    ):
        for k, v in dct2.items():
            if k in dct1:
                dct1[k] = tuple(a + b for a, b in zip(dct1[k], v))
            else:
                dct1[k] = v

    # ranges from -1 to 1. 0 indicates no correlation.
    def _calc_mcc(self, tp, tn, fp, fn):
        return (tp*tn - fp*fn) / (((tp+fp)(tp+fn)(tn+fp)(tn+fn)) ** .5)

    def _select_features(self, grams: list) -> None:
        for elem in grams:
            total = elem[1][0]
            # false positive
            present_not_phishing = elem[1][1]
            # true positive
            present_phishing = total - present_not_phishing
            # true negative
            not_present_not_phishing = self.num_not_phishing - present_not_phishing
            # false negative
            not_present_phishing = self.num_phishing - present_phishing

            res = abs(self._calc_mcc(
                present_phishing,
                not_present_not_phishing,
                present_phishing,
                not_present_phishing
            ))

            if res >= self.threshold:
                self.space.add(elem)
                self.selected += 1
                if self.selected == self.requested:
                    break


class FlexibleGramExtractor:
    fe: FeatureExtractor
    fs: FeatureSelector
    selection_done: bool
    gram_space: set

    def __init__(self, paths, gram_size, requested_grams, threshold) -> None:
        self.fe = FeatureExtractor(paths)
        self.fs = FeatureSelector(gram_size, requested_grams, threshold)
        self.selection_done = False

    def extract(self) -> None:
        self.fe.extract(self._prep, self._extract_features)

    def export(self, path: str) -> None:
        self.fe.export(path)

    def _prep(self, chunks, num_chunks) -> None:
        if self.selection_done == False:
            self.gram_space = self.fs.select(chunks, num_chunks)
            self.selection_done = True

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(_process_row, axis=1)

    def _process_row(self, row: pd.Series) -> pd.Series:
        url = row["url"]
        features = {
                "url"   : row["url"],
                "label" : row["label"]
                }

        for elem in self.gram_space:
            features.update({elem:0})
        url_len = len(url)
        if url_len >= self.gram_size:
            for i in range(url_len - self.gram_size):
                key = url[i : i + self.gram_size]
                if key in self.gram_space:
                    features[key] = 1

        return pd.Series(features)

class BigramExtractor:
    fe: FeatureExtractor
    gram_size: int

    def __init__(self, paths) -> None:
        self.fe = FeatureExtractor(paths)

    def extract(self) -> None:
        self.fe.extract(None, self._extract_features)

    def export(self, path: str) -> None:
        self.fe.export(path)

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(self._process_row, axis=1)

    def _process_row(self, row: pd.Series) -> pd.Series:
        url = row["url"]
        features = {
                "url"   : row["url"],
                "label" : row["label"]
                }

        bigram_presence = self._bigram_extract(url)

        for i, j in enumerate(bigram_presence):
            idx1 = i // CHAR_SPACE_LEN
            idx2 = i % CHAR_SPACE_LEN
            big = CHAR_SPACE[idx1] + CHAR_SPACE[idx2]
            features.update({big:j})

        return pd.Series(features)

    def _bigram_extract(self, url: str) -> List[int]:
        url_len = len(url)
        total_bigrams = url_len - 1
        presence = [0] * (self.CHAR_SPACE_LEN**2)

        for i in range(total_bigrams):
            idx = CHAR_INDEX[url[i]] * CHAR_SPACE_LEN + CHAR_INDEX[url[i + 1]]
            presence[idx] = 1


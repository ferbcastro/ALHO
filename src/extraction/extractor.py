from multiprocessing import Pool, cpu_count
from urllib.parse import urlparse
from typing import *

import pandas as pd
import math
import sys
import re

class FeatureExtractor:
    org_df: pd.DataFrame
    mod_df: pd.DataFrame

    char_space = string.printable[:-6] # printable characters except whitespaces
    char_space_len = len(char_space)
    char_index = {c: i for i, c in enumerate(char_space)}

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
        url = "".join(char for char in url if char in self.char_space)

        if (scheme := urlparse(url).scheme):
            url = re.sub(f"^{scheme}://", "", url)

        return url

class FlexibleGramExtractor:
    fe: FeatureExtractor
    requested_grams: int
    gram_size: int
    threshold: float
    gram_space: set
    num_selected_grams: int
    num_not_phishing: int
    num_phishing: int
    selection_done: bool

    def __init__(self, paths, gram_size, requested_grams, threshold) -> None:
        self.fe = FeatureExtractor(paths)

        max_grams = fe.char_space_len ** gram_size
        if requested_grams > max_grams:
            self.requested_grams = max_grams
            print(f'requested_grams set to {num}', self.requested_grams)
        else:
            self.requested_grams = requested_grams

        if threshold < 0 or threshold > 1:
            self.threshold = 0.5
            print(f'threshold set to {num}', self.threshold)
        else:
            self.threshold = threshold

        self.num_selected_grams = self.num_not_phishing = self.num_phishing = 0
        self.selection_done = False

    def extract(self) -> None:
        self.fe.extract(self._prep, self._extract_features)

    def export(self, path: str) -> None:
        self.fe.export(path)

    def statistics(self) -> None:
        print(f'Number of selected grams [{num}]', self.num_selected_grams)
        print(f'Threshold set to [{num}]', self.threshold)

    def _prep(self, chunks, num_chunks) -> None:
        if self.selection_done == True:
            return

        with Pool(num_chunks) as pool:
            results = pool.map(self._build_dictionary, chunks)

        total_grams_dict = dict()
        for d in results:
            self._merge_dict(total_grams_dict, d)

        view = total_grams_dict.items()
        sorted_grams_list = sorted(view, lambda it : it[1][0], reverse=True)
        self._select_features(sorted_grams_list)
        self.selection_done = True

    def _build_dictionary(df: pd.DataFrame) -> dict[str : tuple[int, int]]:
        dct = dict()
        df.apply(lambda row : self._build_dict_process_row(row, dct), axis = 1)
        return dct

    def _build_dict_process_row(row: pd.Series, dct) -> None:
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
        dct1: dict[str, tuple[int, int]],
        dct2: dict[str, tuple[int, int]]
    ):
        for k, v in dct2.items():
            if k in dct1:
                dct1[k] = tuple(a + b for a, b in zip(dct1[k], v))
            else:
                dct1[k] = v

    # ranges from -1 to 1. 0 indicates no correlation.
    def _calc_mcc(tp, tn, fp, fn):
        return (tp*tn - fp*fn) / (((tp+fp)(tp+fn)(tn+fp)(tn+fn)) ** .5)

    def _select_features(grams: list) -> None:
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
                self.gram_space.add(elem)
                self.num_selected_grams += 1
                if self.num_selected_grams == self.requested_grams:
                    break

    def _extract_features(df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(_process_row, axis=1)

    def _process_row(row: pd.Series) -> pd.Series:
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

    def _extract_features(df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(self._process_row, axis=1)

    def _process_row(row: pd.Series) -> pd.Series:
        url = row["url"]
        features = {
                "url"   : row["url"],
                "label" : row["label"]
                }

        bigram_presence = self._bigram_extract(url)

        for i, j in enumerate(bigram_presence):
            idx1 = i // self.fe.char_space_len
            idx2 = i % self.fe.char_space_len
            big = self.fe.char_space[idx1] + self.fe.char_space[idx2]
            features.update({big:j})

        return pd.Series(features)

    def _bigram_extract(url: str) -> List[int]:
        url_len = len(url)
        total_bigrams = url_len - 1
        presence = [0] * (self.char_space_len**2)

        for i in range(total_bigrams):
            idx = self.char_index[url[i]] * char_space_len + \
                self.char_index[url[i + 1]]
            presence[idx] = 1


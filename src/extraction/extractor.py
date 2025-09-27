import sys
from multiprocessing import Pool, cpu_count
from urllib.parse import urlparse
from typing import *

import pandas as pd
import math
import re
import string

CHAR_SPACE = string.printable[:-6] # printable characters except whitespaces
CHAR_SPACE_LEN = len(CHAR_SPACE)
CHAR_INDEX = {c: i for i, c in enumerate(CHAR_SPACE)}

URL_FIELD = 'URL'
LABEL_FIELD = 'label'
FREQ_FIELD = 'frequency'
GRAM_NAME_FIELD = 'gram_names'
CORR_FIELD = 'correlation'
PPV_FIELD = 'ppv'
NPV_FIELD = 'npv'
SENS_FIELD = 'sensitivity'
SPEC_FIELD = 'specificity'
FSCORE_FIELD = 'fscore'
PHISHING_LABEL = 0
NOT_PHISHING_LABEL = 1

# Strip scheme and characters outside CHAR_SPACE
def strip_url(url: str) -> str:
    url = "".join(char for char in url if char in CHAR_SPACE)

    if (scheme := urlparse(url).scheme):
        url = re.sub(f"^{scheme}://", "", url)

    return url

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
        df['url'] = df['url'].apply(strip_url)
        org_df = mod_df = df

    def extract(self, prep_func, extraction_func) -> None:
        n_procs = cpu_count()
        chunk_size = math.ceil(org_df.shape[0] / n_procs)
        chunks = []
        for i in range(n_procs):
            start_index = i * chunk_size
            end_index = (i + 1) * chunk_size
            df_chunk = org_df.iloc[start_index:end_index]
            chunks.append(df_chunk)
        n_chunks = n_procs
        if prep_func:
            prep_func(chunks, n_chunks) # any preprocessing needed
        with Pool(n_chunks) as pool:
            results = pool.map(extraction_func, chunks)
        self.mod_df = pd.concat(results, ignore_index=True)

    def export(self, path: str) -> None:
        self.mod_df.to_csv(path, index=False)

class FeatureSelector:
    num_not_phishing: int
    num_phishing: int
    selected: int
    requested: int
    gram_size: int
    space: set

    csv_col_names = [GRAM_NAME_FIELD, \
                FREQ_FIELD, \
                CORR_FIELD, \
                PPV_FIELD, \
                NPV_FIELD, \
                SENS_FIELD, \
                SPEC_FIELD, \
                FSCORE_FIELD]

    def __init__(self, gram_size, requested) -> None:
        assert gram_size > 0
        self.gram_size = gram_size
        max_f = CHAR_SPACE_LEN ** self.gram_size
        assert requested <= max_f
        self.requested = requested

        self.num_not_phishing = self.num_phishing = self.selected = 0
        self.space = set()
        self.features_info = []

    def select(self, df : pd.DataFrame) -> set:
        total_grams_dict = self._build_dictionary(df)
        self._select_features(total_grams_dict)
        return self.space

    def statistics(self) -> None:
        print('Printing statistics')
        print(f'Number of selected grams [{self.selected}]')
        print(f'Threshold set to [{self.threshold}]')

    def dump_info(self) -> None:
        df = pd.DataFrame(data = self.features_info, columns = self.csv_col_names)
        df = df.astype({
            GRAM_NAME_FIELD: str,
            FREQ_FIELD: int,
        })
        df.to_csv(f'features_info_{self.gram_size}.csv', index = False)
        print(f'Zero ppv count [{self.zero_ppv_count}]')
        print(f'Zero npv count [{self.zero_npv_count}]')
        print(f'Zero sens count [{self.zero_sens_count}]')
        print(f'Zero spec count [{self.zero_spec_count}]')

    def _build_dictionary(self, df: pd.DataFrame) -> dict[str : list]:
        dct = {}
        for _, row in df.iterrows():
            url = row[URL_FIELD]
            label = row[LABEL_FIELD]
            url_len = len(url)

            if url_len < self.gram_size:
                continue

            if label == NOT_PHISHING_LABEL:
                self.num_not_phishing += 1
            else:
                self.num_phishing += 1
            aux_set = set()
            for i in range(url_len - self.gram_size):
                key = url[i : i + self.gram_size]
                if key not in aux_set:
                    if key not in dct:
                        dct.update({key : [1, 0]})
                    dct[key][0] += 1
                    aux_set.add(key)
                    if label == NOT_PHISHING_LABEL:
                        dct[key][1] += 1
        return dct

    # ranges from -1 to 1. 0 indicates no correlation.
    def _calc_mcc(self, tp, tn, fp, fn):
        return (tp*tn - fp*fn) / (((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** .5)

    def _calc_ppv(self, tp, fp):
        return tp / (tp+fp)

    def _calc_npv(self, tn, fn):
        return tn / (tn+fn)

    def _calc_sens(self, tp, fn):
        return tp / (tp+fn)

    def _calc_spec(self, tn, fp):
        return tn / (tn+fp)

    def _calc_fscore(self, precision, recall):
        return (2 * precision * recall) / (precision + recall)

    def _select_features(self, grams: dict) -> None:
        self.zero_ppv_count = 0
        self.zero_npv_count = 0
        self.zero_sens_count = 0
        self.zero_spec_count = 0
        gram_and_corr = []
        for elem in grams.items():
            total = elem[1][0]
            present_not_phishing = elem[1][1]
            present_phishing = total - present_not_phishing
            not_present_not_phishing = self.num_not_phishing - present_not_phishing
            not_present_phishing = self.num_phishing - present_phishing
            corr = abs(self._calc_mcc(
                present_phishing,
                not_present_not_phishing,
                present_not_phishing,
                not_present_phishing
            ))
            ppv = self._calc_ppv(
                present_phishing,
                present_not_phishing
            )
            if ppv == 0:
                self.zero_ppv_count += 1
            npv = self._calc_npv(
                not_present_not_phishing,
                not_present_phishing
            )
            if npv == 0:
                self.zero_npv_count += 1
            sens = self._calc_sens(
                present_phishing,
                not_present_phishing
            )
            if sens == 0:
                self.zero_sens_count += 1
            spec = self._calc_spec(
                not_present_not_phishing,
                present_not_phishing
            )
            if spec == 0:
                self.zero_spec_count += 1
            fscore = self._calc_fscore(
                ppv,
                sens
            )
            gram_and_corr.append([elem[0], total, corr, ppv, npv, sens, spec, fscore])

        sorted_corr = sorted(gram_and_corr, key = lambda it : it[2], reverse = True)
        for i in range(self.requested):
            self.features_info.append(sorted_corr[i])
            self.space.add(sorted_corr[i][0])

class FlexibleGramExtractor:
    fe: FeatureExtractor
    fs: FeatureSelector
    selection_done = False
    gram_space = set()

    def __init__(self, paths, gram_size, requested_grams, threshold) -> None:
        self.fe = FeatureExtractor(paths)
        self.fs = FeatureSelector(gram_size, requested_grams, threshold)

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
        presence = [0] * (CHAR_SPACE_LEN**2)

        for i in range(total_bigrams):
            idx = CHAR_INDEX[url[i]] * CHAR_SPACE_LEN + CHAR_INDEX[url[i + 1]]
            presence[idx] = 1

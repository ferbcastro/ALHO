from multiprocessing import Pool, cpu_count
from urllib.parse import urlparse

import math
import re
import string
import pandas as pd

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
        self.org_df = self.mod_df = df

    def extract(self, prep_func, extraction_func) -> None:
        n_procs = cpu_count()
        chunk_size = math.ceil(self.org_df.shape[0] / n_procs)
        chunks = []
        for i in range(n_procs):
            start_index = i * chunk_size
            end_index = (i + 1) * chunk_size
            df_chunk = self.org_df.iloc[start_index:end_index]
            chunks.append(df_chunk)
        n_chunks = n_procs
        if prep_func:
            prep_func(self.org_df) # any preprocessing needed
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
    label_phishing: int
    label_legitimate: int
    seleted_only_on_freq: bool

    cols_1 = [GRAM_NAME_FIELD, \
            FREQ_FIELD, \
            CORR_FIELD, \
            PPV_FIELD, \
            NPV_FIELD, \
            SENS_FIELD, \
            SPEC_FIELD]

    cols_2 = [GRAM_NAME_FIELD, FREQ_FIELD]

    def __init__(self, gram_size, requested) -> None:
        assert gram_size > 0
        self.gram_size = gram_size
        max_f = CHAR_SPACE_LEN ** self.gram_size
        assert requested <= max_f
        self.requested = requested

        self.num_not_phishing = self.num_phishing = self.selected = 0
        self.space = set()
        self.features_info = []

    def select(self, df : pd.DataFrame, l_phishing, l_legitimate, force_freq_sel) -> set:
        self.label_phishing = l_phishing
        self.label_legitimate = l_legitimate
        total_grams_dict = self._build_dictionary(df)
        if self.num_not_phishing == 0 or self.num_phishing == 0 or force_freq_sel:
            print('selecting based on frequency only...')
            self._select_features_on_freqs(total_grams_dict)
            self.seleted_only_on_freq = True
        else:
            print('selecting based on frequency and ppv...')
            self._select_features_on_ppv_freqs(total_grams_dict)
            self.seleted_only_on_freq = False
        return self.space

    def statistics(self) -> None:
        print('printing statistics')
        print(f'number of selected grams [{self.selected}]')

    def dump_info(self, name : str) -> None:
        if self.seleted_only_on_freq:
            df = pd.DataFrame(data = self.features_info, columns = self.cols_2)
        else:
            df = pd.DataFrame(data = self.features_info, columns = self.cols_1)
        df = df.astype({
            GRAM_NAME_FIELD: str,
            FREQ_FIELD: int
        })

        df.to_csv(f'features_{name}_{self.gram_size}.csv', index = False)

    def _build_dictionary(self, df: pd.DataFrame) -> dict[str : list]:
        dct = {}
        for _, row in df.iterrows():
            url = row[URL_FIELD]
            label = row[LABEL_FIELD]
            url_len = len(url)

            if url_len < self.gram_size:
                continue

            if label == self.label_legitimate:
                self.num_not_phishing += 1
            else:
                self.num_phishing += 1
            aux_set = set()
            for i in range(url_len - self.gram_size):
                key = url[i : i + self.gram_size]
                if key.isdigit():
                    continue
                if key not in aux_set:
                    if key not in dct:
                        dct.update({key : [0, 0]})
                    dct[key][0] += 1
                    aux_set.add(key)
                    if label == self.label_legitimate:
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

    def _select_features_on_freqs(self, grams: dict) -> None:
        view = grams.items()
        sorted_arr_freqs = sorted(view, key = lambda it : it[1][0], reverse = True)
        for elem in sorted_arr_freqs[:self.requested]:
            self.features_info.append([elem[0], elem[1][0]])
            self.space.add(elem[0])

    def _select_features_on_ppv_freqs(self, grams: dict) -> None:
        gram_and_corr = []
        for elem in grams.items():
            total = elem[1][0]
            fp = present_not_phishing = elem[1][1]
            tp = present_phishing = total - present_not_phishing
            tn = not_present_not_phishing = self.num_not_phishing - present_not_phishing
            fn = not_present_phishing = self.num_phishing - present_phishing

            # print(not_present_not_phishing, not_present_phishing, elem)

            if ((tp+fp) == 0 or (tp+fn) == 0 or (tn+fp) == 0 or (tn+fn) == 0):
                print(f'Skiping n-gram {elem}')
                continue

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
            npv = self._calc_npv(
                not_present_not_phishing,
                not_present_phishing
            )
            sens = self._calc_sens(
                present_phishing,
                not_present_phishing
            )
            spec = self._calc_spec(
                not_present_not_phishing,
                present_not_phishing
            )
            gram_and_corr.append([elem[0], total, corr, ppv, npv, sens, spec])

        #gram_and_corr = sorted(gram_and_corr, key = lambda it : it[2], reverse = True)
        #self.features_info = gram_and_corr[:self.requested]
        sorted_arr_ppv = sorted(gram_and_corr, key = lambda it : it[3], reverse = True)
        sorted_arr_ppv_sub = sorted_arr_ppv[:self.requested * 2]
        sorted_arr_ppv_npv = sorted(sorted_arr_ppv_sub, key = lambda it : it[4], reverse = True)
        sorted_arr_ppv_npv_sub = sorted_arr_ppv_npv[:self.requested]
        sorted_arr_freq_sub = sorted(sorted_arr_ppv_npv_sub, key = lambda it : it[1], reverse = True)
        self.features_info = sorted_arr_freq_sub[:200]
        for elem in self.features_info:
            self.space.add(elem[0])

class FlexibleGramExtractor:
    fe: FeatureExtractor
    fs: FeatureSelector
    selection_done = False
    gram_space = set()

    def __init__(self, paths, gram_size, requested_grams) -> None:
        self.fe = FeatureExtractor(paths)
        self.fs = FeatureSelector(gram_size, requested_grams)

    def extract(self) -> None:
        self.fe.extract(self._prep, self._extract_features)

    def export(self, path: str) -> None:
        self.fe.export(path)

    def _prep(self, org_df) -> None:
        if not self.selection_done:
            self.gram_space = self.fs.select(org_df)
            self.selection_done = True

    def _process_row(self, row: pd.Series) -> pd.Series:
        url = row["url"]
        features = {
                "url"   : row["url"],
                "label" : row["label"]
                }

        for elem in self.gram_space:
            features.update({elem:0})
        url_len = len(url)
        if url_len >= self.fs.gram_size:
            for i in range(url_len - self.fs.gram_size):
                key = url[i : i + self.fs.gram_size]
                if key in self.gram_space:
                    features[key] = 1

        return pd.Series(features)

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(self._process_row, axis=1)


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

    def _bigram_extract(self, url: str) -> list[int]:
        url_len = len(url)
        total_bigrams = url_len - 1
        presence = [0] * (CHAR_SPACE_LEN**2)

        for i in range(total_bigrams):
            idx = CHAR_INDEX[url[i]] * CHAR_SPACE_LEN + CHAR_INDEX[url[i + 1]]
            presence[idx] = 1

        return presence

import pandas as pd
import string

URL_FIELD = 'URL'
LABEL_FIELD = 'label'
FREQ_FIELD = 'frequency'
GRAM_NAME_FIELD = 'gram_names'
CORR_FIELD = 'correlation'
PPV_FIELD = 'ppv'
NPV_FIELD = 'npv'
SENS_FIELD = 'sensitivity'
SPEC_FIELD = 'specificity'
#FSCORE_FIELD = 'fscore'

CHAR_SPACE = string.printable[:-6] # printable characters except whitespaces
CHAR_SPACE_LEN = len(CHAR_SPACE)
CHAR_INDEX = {c: i for i, c in enumerate(CHAR_SPACE)}

cols_1 = [GRAM_NAME_FIELD, FREQ_FIELD, CORR_FIELD, PPV_FIELD, NPV_FIELD, SENS_FIELD, SPEC_FIELD]
cols_2 = [GRAM_NAME_FIELD, FREQ_FIELD]

def calc_mcc(tp, tn, fp, fn):
    return (tp*tn - fp*fn) / (((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** .5)

def calc_ppv(tp, fp):
    return tp / (tp+fp)

def calc_npv(tn, fn):
    return tn / (tn+fn)

def calc_sens(tp, fn):
    return tp / (tp+fn)

def calc_spec(tn, fp):
    return tn / (tn+fp)

def calc_fscore(precision, recall):
    return (2 * precision * recall) / (precision + recall)

class FeatureSelector:
    num_not_phishing: int
    num_phishing: int
    selected: int
    requested: int
    gram_size: int
    label_phishing: int
    label_legitimate: int
    seleted_only_on_freq: bool

    def __init__(self, gram_size, requested) -> None:
        assert gram_size > 0
        self.gram_size = gram_size
        max_f = CHAR_SPACE_LEN ** self.gram_size
        assert requested <= max_f
        self.requested = requested

        self.num_not_phishing = self.num_phishing = self.selected = 0
        self.features_info = []

    def select(self, df : pd.DataFrame, l_phishing, l_legitimate):
        self.label_phishing = l_phishing
        self.label_legitimate = l_legitimate
        total_grams_dict = self._build_dictionary(df)
        if self.num_not_phishing == 0 or self.num_phishing == 0:
            print('selecting based on frequency only...')
            self._select_features_on_freqs(total_grams_dict)
            self.seleted_only_on_freq = True
        else:
            print('selecting based on frequency and ppv...')
            self._select_features_on_ppv_freqs(total_grams_dict)
            self.seleted_only_on_freq = False

    def statistics(self) -> None:
        print('printing statistics')
        print(f'number of selected grams [{self.selected}]')

    def dump_info(self, name : str) -> None:
        global cols1
        df = pd.DataFrame(data = self.features_info, columns = cols_1)
        df = df.astype({
            GRAM_NAME_FIELD: str,
            FREQ_FIELD: int
        })

        df.to_csv(f'{name}', index = False)

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

            if ((tp+fp) == 0 or (tp+fn) == 0 or (tn+fp) == 0 or (tn+fn) == 0):
                print(f'Skiping n-gram {elem}')
                continue

            corr = abs(calc_mcc(
                present_phishing,
                not_present_not_phishing,
                present_not_phishing,
                not_present_phishing
            ))
            ppv = calc_ppv(
                present_phishing,
                present_not_phishing
            )
            npv = calc_npv(
                not_present_not_phishing,
                not_present_phishing
            )
            sens = calc_sens(
                present_phishing,
                not_present_phishing
            )
            spec = calc_spec(
                not_present_not_phishing,
                present_not_phishing
            )
            gram_and_corr.append([elem[0], total, corr, ppv, npv, sens, spec])

        sorted_arr_ppv = sorted(gram_and_corr, key = lambda it : it[3], reverse = True)
        sorted_arr_ppv_sub = sorted_arr_ppv[:self.requested * 4]
        sorted_arr_ppv_npv = sorted(sorted_arr_ppv_sub, key = lambda it : it[4], reverse = True)
        sorted_arr_ppv_npv_sub = sorted_arr_ppv_npv[:self.requested * 2]
        sorted_arr_freq_sub = sorted(sorted_arr_ppv_npv_sub, key = lambda it : it[1], reverse = True)
        self.features_info = sorted_arr_freq_sub[:self.requested]

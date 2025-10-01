from multiprocessing import Pool, cpu_count
import pandas as pd

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
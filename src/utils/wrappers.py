import pandas as pd

def open(path, use = None, types = None) -> pd.DataFrame:
    df = pd.read_csv(path, usecols = use, dtype = types)
    return df

def concat(paths, use = None, types = None) -> pd.DataFrame:
    df1 = pd.read_csv(paths[0], usecols = use, dtype = types)
    frames = [df1]
    for path in paths[1:]:
        tmp = pd.read_csv(path, usecols = use, dtype = types)
        frames.append(tmp)
    df = pd.concat(frames, ignore_index = True)
    return df

def export(df, path, idx = False) -> None:
    df.to_csv(path, index = idx)
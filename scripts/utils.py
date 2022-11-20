import pandas as pd

def get_data_frame(*paths):
        paths = list(sum(paths, []))
        print(paths)
        dfs = list(map(lambda path: pd.read_csv(path), paths))
        df = pd.concat(dfs, join='outer', axis=1)
        df = df.loc[:,~df.columns.duplicated()].copy()
        return df
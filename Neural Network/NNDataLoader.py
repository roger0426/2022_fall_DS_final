from pathlib import Path
from typing import *

import numpy as np
import pandas as pd


def _get_df(path: Union[Path, str]):
    df: pd.DataFrame = pd.read_pickle(path)
    df = df.sort_values(by=['ncodpers', 'fecha_dato'])
    for n in df.columns:
        print(n, "\t", df[n].iloc[0])


def _gen_debug_df(path):
    df: pd.DataFrame = pd.read_pickle(path)
    df = df.iloc[:10000]
    df.to_pickle('./debug.pkl')


def main():
    # _gen_debug_df('../data/train_preprocessed_v4.pkl')
    _get_df('./debug.pkl')


if __name__ == '__main__':
    main()

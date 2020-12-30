import pandas as pd
import numpy as np
def columns_count(df,key):
    print(df[key].isna().sum() / len(df))
    if len(df[key].unique())<5:
        df[key].value_counts(normalize=True).plot(kind='bar',figsize=(6, 8));
    else:
        df[key].value_counts(normalize=True).plot(kind='barh',figsize=(10, 20));
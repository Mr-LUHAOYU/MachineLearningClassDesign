import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'

path = "data/merged_data"

for i in range(1, 17):
    print(i)
    df = pd.read_csv(f"{path}/data_{i:03}.csv")
    df.head()
    # find missing values
    missing_values = df.isna()
    missing_positions = missing_values[missing_values].stack().index.tolist()
    # fill missing values
    for row, col in missing_positions:
        source_row = (row + 345600 - 1) % 345600
        df.loc[row, col] = df.loc[source_row, col]
    # normalize data
    columns_to_normalize = df.columns.difference(['datetime', 'glucose'])
    df[columns_to_normalize] = (
            (df[columns_to_normalize] - df[columns_to_normalize].min())
            /
            (df[columns_to_normalize].max() - df[columns_to_normalize].min())
    )
    df.head()
    df.to_csv(f"data/cleaned_data/data_{i:03}_filled.csv", index=False)

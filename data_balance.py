import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE


def balance(df_in: pd.DataFrame, threshold: float, target_variable: str = 'glucose') -> pd.DataFrame:
    df = df_in[df_in[target_variable] > 0]
    df_rest = df_in[df_in[target_variable] <= 0]
    virtual_label = np.where(df[target_variable] > threshold, 1, 0)
    smote = SMOTE()
    df, _ = smote.fit_resample(df, virtual_label)
    df_balanced = pd.concat([df, df_rest]).reset_index(drop=True)
    return df_balanced


def save(num: int):
    df = pd.read_csv(f'data/extraction_data_reduced/data_{num:03}.csv')
    df['time'] = df.index
    df['glucose'] = df['glucose'].apply(
        lambda x: x + np.random.normal(0, 0.01) if x > 0.2 else np.nan
    )
    df.dropna(inplace=True)
    for i in [0.7 * df['glucose'].max()]:
        try:
            df_balanced = balance(df, i)
        except ValueError:
            df_balanced = df
            print(f'Data {num:03} is unexpectedly balanced on {i}.')
    df_balanced.to_csv(f'data/balanced_data/data_{num:03}.csv', index=False)


def main():
    for i in range(1, 17):
        save(i)
        print(f'Data {i:03} is balanced.')


np.random.seed(607)
main()

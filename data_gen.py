import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from math import floor, ceil
from IPython.display import display

pth = "data"
start_time = datetime.datetime.strptime("00:00:00.000", '%H:%M:%S.%f')
end_time = datetime.datetime.strptime("23:59:59.750", '%H:%M:%S.%f')


class DataGenerator(object):
    def __init__(self, num: str, ):
        self.num = num

    def read_data(self) -> dict[str, pd.DataFrame | pd.Series]:
        df_dict = dict()
        info_list = ['ACC', "BVP", "EDA", "HR", "TEMP", "IBI"]
        # num = "001"
        for info in info_list:
            df = pd.read_csv(f"{pth}/{self.num}/{info}_{self.num}.csv")
            df_dict[info] = df
        df = pd.read_csv(f"{pth}/{self.num}/Dexcom_{self.num}.csv")
        df = df[['Timestamp (YYYY-MM-DDThh:mm:ss)', 'Glucose Value (mg/dL)']]
        df.dropna(inplace=True)
        df.columns = ['datetime', 'glucose']
        df_dict['glucose'] = df
        # display(df)
        return df_dict

    @staticmethod
    def get_time_series() -> pd.DataFrame | pd.Series:
        data = pd.DataFrame(list(range(345600)), columns=['datetime'])
        return data

    @staticmethod
    def preprocess(df: pd.DataFrame, ms: bool) -> pd.DataFrame:
        def upper_bound(time):
            t = time.split()[-1]
            if ms:
                t = datetime.datetime.strptime(t, '%H:%M:%S.%f')
            else:
                t = datetime.datetime.strptime(t, '%H:%M:%S')
            x = (t - start_time).total_seconds() * 4
            return ceil(x)

        df['datetime'] = df['datetime'].apply(lambda x: upper_bound(x))
        df = df.groupby('datetime').mean().reset_index()
        return df

    @staticmethod
    def interpolate(data: pd.Series):
        ddate = pd.concat([data, data], axis=0)
        ddate = ddate.interpolate(method='linear')
        data1, data2 = ddate[:data.shape[0]], ddate[data.shape[0]:]
        data = data1.combine_first(data2)
        return data

    def preprocess_1(self, df: pd.DataFrame, data: pd.DataFrame, ms: bool):
        df = self.preprocess(df, ms)
        data = pd.merge(data, df, on='datetime', how='left')
        for col in df.columns:
            if col == "datetime":
                continue
            data[col] = data[col].apply(lambda x: np.mean(x) if x else None)
            # if col == "glucose":
            data[col] = self.interpolate(data[col])
            # else:
            #     data[col] = data[col].ffill()
        return data

    def gen(self, write: bool = False) -> pd.DataFrame:
        df_dict = self.read_data()
        data = self.get_time_series()
        for info in ["EDA", "TEMP", "IBI"]:
            data = self.preprocess_1(df_dict[info], data, True)
            print(f"{info} data preprocessed successfully.")
        for info in ["ACC", "BVP", "HR", "glucose"]:
            data = self.preprocess_1(df_dict[info], data, False)
            print(f"{info} data preprocessed successfully.")
        self.normalize(data)
        if write:
            data.to_csv(f"{pth}/cleaned_data/data_{self.num}.csv", index=False)
        return data

    @staticmethod
    def normalize(df: pd.DataFrame):
        columns_to_normalize = df.columns.difference(['datetime', 'glucose'])
        df[columns_to_normalize] = (
                (df[columns_to_normalize] - df[columns_to_normalize].min())
                /
                (df[columns_to_normalize].max() - df[columns_to_normalize].min())
        )


def draw_all(write: bool = False):
    fig, axs = plt.subplots(4, 4, figsize=(15, 15))
    for i in range(1, 17):
        num = f"{i:03}"
        data_gen = DataGenerator(num)
        all_data = data_gen.gen(write=write)
        glucose = all_data['glucose'][::1200]
        x = all_data['datetime'][::1200].apply(lambda x: x / 4 / 60 / 60)

        row = (i - 1) // 4
        col = (i - 1) % 4

        axs[row, col].plot(x, glucose)
        axs[row, col].set_title(f"Plot {i}")

    plt.tight_layout()
    plt.show()


def main():
    # draw_all(write=True)
    for i in range(1, 17):
        data_gen = DataGenerator(f"{i:03}")
        data_gen.gen(True)


if __name__ == "__main__":
    main()

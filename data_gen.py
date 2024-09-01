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

    def read_data(self):
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
    def get_time_series():
        data = pd.DataFrame(list(range(345600)), columns=['datetime'])
        return data

    @staticmethod
    def preprocess(df: pd.DataFrame, ms: bool):
        def lower_bound(time):
            t = time.split()[-1]
            if ms:
                t = datetime.datetime.strptime(t, '%H:%M:%S.%f')
            else:
                t = datetime.datetime.strptime(t, '%H:%M:%S')
            x = (t - start_time).total_seconds() * 4
            return ceil(x)

        df['datetime'] = df['datetime'].apply(lambda x: lower_bound(x))
        df = df.groupby('datetime').mean().reset_index()
        return df

    def preprocess_1(self, df: pd.DataFrame, data: pd.DataFrame, ms: bool):
        df = self.preprocess(df, ms)
        data = pd.merge(data, df, on='datetime', how='left')
        for col in df.columns:
            if col == "datetime":
                continue
            data[col] = data[col].apply(lambda x: np.mean(x) if x else None)
            # if col == "glucose":
            data[col] = data[col].interpolate(method='linear')
            # else:
            #     data[col] = data[col].ffill()
        return data

    def gen(self):
        df_dict = self.read_data()
        data = self.get_time_series()
        for info in ["EDA", "TEMP", "IBI"]:
            data = self.preprocess_1(df_dict[info], data, True)
            print(f"{info} data preprocessed successfully.")
        for info in ["ACC", "BVP", "HR", "glucose"]:
            data = self.preprocess_1(df_dict[info], data, False)
            print(f"{info} data preprocessed successfully.")
        data.to_csv(f"{pth}/merged_data/data_{self.num}.csv", index=False)


if __name__ == '__main__':
    for i in range(1, 17):
        dg = DataGenerator(f"{i:03}")
        dg.gen()
        print(f"Data for patient {i} generated successfully.")

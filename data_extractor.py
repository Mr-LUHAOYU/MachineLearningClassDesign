# def glucose_to_possible(
#         df: pd.DataFrame | pd.Series, 
#         window_size: int, 
#         step_size: int,
# ) -> pd.Series:
#     if type(df) == pd.Series:
#         df = pd.DataFrame(df)
#     p: list[float] = [1.0]
#     f: list[float] = [1.0]
#     def funcp(series: pd.Series) -> float:
#         p.append(1)
#         f1, f2 = f[-2], f[-1]
#         p1, p2 = p[-2], p[-1]
#         delta = np.random.normal(0, 0.05, 1)[0]
#         p2 = f2 + delta
#         if p2 < 0:
#             p2 = 0
#         if p2 > 1:
#             p2 = 1
#         p[-1] = p2
#         return p2
#         
#     def funcf(series: pd.Series) -> float:
#         s = (series >= 140).sum()
#         f.append(s / window_size)
#         return f[-1]
#     
#     for start in range(0, len(df) - window_size + 1, step_size):
#         sub_df = df.iloc[start:start + window_size]
#         sub_df.apply(funcf, axis=0)
#         sub_df.apply(funcp, axis=0)
#     return pd.Series(p[1::], name='glucose')
# 
# def save():
#     for i in range(1, 17):
#         df = pd.read_csv(f'data/cleaned_data/data_{i:03}.csv')
#         window_size = 4 * 60 * 5
#         step_size = 4 * 60
#         glucose = glucose_to_possible(
#             df['glucose'], window_size, step_size
#         )
#         glucose.to_csv(f'data/extraction_data/data_{i:03}.csv', index=True)
# 
# 
# save()



# import sys
# import pandas as pd
# import numpy as np
# from sklearn.metrics import mutual_info_score
# import warnings
# warnings.filterwarnings('ignore')
#
# def calculate_correlation(series1, series2):
#     mutual_info = mutual_info_score(series1.to_numpy(), series2.to_numpy())
#     print(f"互信息: {mutual_info}")
#
# cur_col = ['datetime', ' eda', ' temp', ' acc_x', ' acc_y', ' acc_z', ' hr'][6]
#
#
# def sliding_window_transform(
#         df: pd.DataFrame | pd.Series,
#         window_size: int,
#         step_size: int,
#         func: callable
# ) -> pd.Series:
#     if type(df) == pd.Series:
#         df = pd.DataFrame(df)
#     results = []
#     for start in range(0, len(df) - window_size + 1, step_size):
#         sub_df = df.iloc[start:start + window_size]
#         transformed_sub_df = sub_df.apply(func, axis=0).to_numpy()[0]
#         results.append(transformed_sub_df)
#     result_df = pd.Series(results, name=feature_name)
#     return result_df
#
#
# names = ['mean', 'min', 'max', 'std', 'median', 'pk', 'kurt', ][6]
# feature_name = f"{cur_col}_{names}"
# print(f"current feature: {feature_name}")
#
# def feature_func(series: pd.Series) -> float | int:
#     match names:
#         case'mean':
#             return series.mean()
#         case'min':
#             return series.min()
#         case'max':
#             return series.max()
#         case'std':
#             return series.std()
#         case'median':
#             return series.median()
#         case'pk':
#             return series.max() - series.min()
#         case'kurt':
#             return series.kurt()
#
#
# def main():
#     df_list = []
#     for i in range(1, 17):
#         df = pd.read_csv(f'data/cleaned_data/data_{i:03}.csv')
#         window_size = 4 * 60 * 5
#         step_size = 4 * 60
#         feature = sliding_window_transform(
#             df[cur_col], window_size, step_size, feature_func
#         )
#         glucose = pd.read_csv(f'data/extraction_data/data_{i:03}.csv')['glucose']
#         calculate_correlation(feature, glucose)
#         df_list.append(feature)
#     s: str = input("need to write? (y/n) ")
#     if s == 'y':
#         for i in range(1, 17):
#             try:
#                 old = pd.read_csv(f"data/extraction_data/data_{i:03}.csv")
#             except:
#                 old = pd.DataFrame()
#             old[feature_name] = df_list[i - 1]
#             old.to_csv(f"data/extraction_data/data_{i:03}.csv", index=False)
#
# main()



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import detrend
from scipy.signal.windows import hann

def draw_fft() -> None:
	# ['datetime', ' eda', ' temp', ' ibi', ' acc_x', ' acc_y', ' acc_z', ' bvp', ' hr', 'glucose']
	for cur_col in ['datetime', ' eda', ' temp', ' ibi', ' acc_x', ' acc_y', ' acc_z', ' bvp', ' hr', 'glucose']:
		fig, axs = plt.subplots(4, 4, figsize=(15, 15))
		for i in range(1, 17):
			df = pd.read_csv(f'data/cleaned_data/data_{i:03}.csv')
			col_centered = df[cur_col] - np.mean(df[cur_col])
			col_detrended = detrend(col_centered)
			window = hann(len(col_detrended))
			col_windowed = col_detrended * window
			fft_values = np.fft.fft(col_windowed)
			n = len(col_windowed)
			freq = np.fft.fftfreq(n, d=0.25)
			amplitude = np.abs(fft_values)
			start, end = 20, n // 2000
			important_frequencies = freq[start:end]
			important_amplitudes = amplitude[start:end]
			dominant_frequency = important_frequencies[np.argmax(important_amplitudes)]
			print(f"Dominant frequency: {dominant_frequency} Hz")
			
			row = (i - 1) // 4
			col = (i - 1) % 4
			
			axs[row, col].plot(important_frequencies, important_amplitudes)
		
		plt.tight_layout()
		plt.savefig(f'Output/{cur_col}_fig.png')
		plt.show()

def draw_corr_matrix() -> None:
	def draw_correlation_matrix(corr_matrix: pd.DataFrame, name: str) -> None:
		plt.figure(figsize=(10 * 2, 8 * 2), dpi=300)
		sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
		plt.title('Correlation Matrix')
		plt.savefig(f'./Output/{name}.png')
		plt.show()
	
	# 计算当前数据集的相关性矩阵
	num = 9
	corr_lim = 0.65
	file_name = f'data_{num:03}.csv'
	folder_path = f'data/extraction_data/'
	data = pd.read_csv(folder_path + file_name, usecols=lambda column: column != 'Unnamed: 0')
	correlation_matrix = data.corr()
	
	# 绘制原始的相关性矩阵热力图
	print("Original correlation matrix:")
	draw_correlation_matrix(correlation_matrix, name=f'correlation_matrix_{num:03}')
	
	# 找出相关性较高的列
	columns_to_drop = set()
	for i in range(len(correlation_matrix.columns)):
		for j in range(i):
			if abs(correlation_matrix.iloc[i, j]) > corr_lim:
				col_name = correlation_matrix.columns[i]
				columns_to_drop.add(col_name)
	data_reduced = data.drop(columns=columns_to_drop)
	
	# 打印相关性较高的列
	print("\nColumns to drop:", columns_to_drop)
	
	# 绘制删除相关性较高的列后的相关性矩阵热力图
	print("\nReduced correlation matrix:")
	draw_correlation_matrix(data_reduced.corr(), name=f'reduced_correlation_matrix_{num:03}')
	
	# 保存删除相关性较高的列后的数据
	new_folder_path = f'data/extraction_data_reduced/'
	for file in os.listdir(folder_path):
		if file.endswith('.csv'):
			now_data = pd.read_csv(folder_path + file, usecols=lambda column: column != 'Unnamed: 0')
			now_data_reduced = now_data.drop(columns=columns_to_drop)
			now_data_reduced.to_csv(new_folder_path + file, index=False)

if __name__ == '__main__':
	...

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.signal.windows import hann

fig, axs = plt.subplots(4, 4, figsize=(15, 15))

for i in range(1, 17):
    df = pd.read_csv(f'data/cleaned_data/data_{i:03}_filled.csv')

    # 去除直流分量
    glucose_centered = df['glucose'] - np.mean(df['glucose'])

    # 去趋势处理
    glucose_detrended = detrend(glucose_centered)

    # 应用窗口函数
    window = hann(len(glucose_detrended))
    glucose_windowed = glucose_detrended * window

    # 执行傅里叶变换
    fft_values = np.fft.fft(glucose_windowed)

    # 计算频率
    n = len(glucose_windowed)
    freq = np.fft.fftfreq(n, d=0.25)  # d 是时间步长

    # 计算幅值（取复数的模）
    amplitude = np.abs(fft_values)

    # 计算主要频率成分（去除直流分量，即频率为0的部分）
    important_frequencies = freq[1:n // 2]
    important_amplitudes = amplitude[1:n // 2]

    # 找到幅值最大的频率
    dominant_frequency = important_frequencies[np.argmax(important_amplitudes)]

    print(f"Dominant frequency: {dominant_frequency} Hz")

    row = (i - 1) // 4
    col = (i - 1) % 4

    # axs[row, col].figure(figsize=(10, 6))
    axs[row, col].plot(important_frequencies, important_amplitudes)
    # axs[row, col].subtitle("Frequency Spectrum")
    # axs[row, col].xlabel("Frequency (Hz)")
    # axs[row, col].ylabel("Amplitude")
    # axs[row, col].yscale('log')  # 使用对数尺度以更好地观察幅值
    # axs[row, col].show()
plt.tight_layout()
plt.show()
# 示例：提取整个序列的主要频率特征
# df['dominant_frequency'] = dominant_frequency

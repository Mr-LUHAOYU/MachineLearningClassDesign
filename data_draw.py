import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.signal.windows import hann

def draw_fft():
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

if __name__ == '__main__':
    ...
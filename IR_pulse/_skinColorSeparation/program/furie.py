import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ハニング窓の関数を定義
def hanning_window(N):
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))
  
# CSVファイルのパスを指定
csv_file_path ='/Users/masayakinefuchi/imageSensing/IR_pulse/_skinColorSeparation/result/gantei5.csv'

# CSVファイルをpandasのDataFrameとして読み込む（列名を指定しない）
df = pd.read_csv(csv_file_path, header=None)

# DataFrameをNumPyの配列に変換してフーリエ変換を行う
data = df.iloc[:, 0].values
sample_rate = 30
duration = 10
N = len(data)  # データ点数
T = 1.0 / 30  # サンプリング間隔

# ハニング窓をデータにかける
window = hanning_window(N)
windowed_data = data * window

F = np.fft.fft(windowed_data)
freq = np.fft.fftfreq(N, d=1/sample_rate)

Amp = np.abs(F)

frequencies = np.fft.fftfreq(N, T)
# フーリエ変換の結果をグラフに表示
plt.plot(freq[1:int(N/2)] , Amp[1:int(N/2)])
plt.xlim(0,3)
plt.xlabel("Frequency [kHz]")
plt.ylabel("Amplitude [#]")
plt.title("FFT test")
plt.show()
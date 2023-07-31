import matplotlib.pyplot as plt
import pandas as pd

# CSVファイルのパスを指定
csv_file_path = '/Users/masayakinefuchi/imageSensing/IR_pulse/_plotPredictPulse/result/gantei5_predict_hemoglobin_bp.csv'

# CSVファイルをpandasのDataFrameとして読み込む
df = pd.read_csv(csv_file_path)

# グラフの描画
plt.figure(figsize=(10, 6))  # グラフのサイズを設定（任意）
plt.plot(df.index, df.iloc[:,0], marker='', linestyle='-')  # X軸とY軸のデータを指定
plt.title('')  # グラフのタイトルを設定（任意）
plt.xlabel('TIme [s]')  # X軸のラベルを設定（任意）
plt.ylabel('Mean ')  # Y軸のラベルを設定（任意）

plt.grid(True)  # グリッド線を表示（任意）

# グラフの表示
plt.show()
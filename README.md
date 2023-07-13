# imageSensing
研究室で利用したプログラムなど


##　色素成分の分離から傾き除去,ピーク検出まで(RGB)
   ### [1]  RGB_Pulse/program/_skinColorSeparation/program/pulseExtract_icexpress_hem.py

      変更する箇所
      ##ROIの位置設定
      x1,y1
      ##ROIの大きさ設定
      width,height
      ##ファイル設定
      INPUT_DIR =''
      OUTPUT_DIR =''
      OUTPUT_FILE = OUTPUT_DIR +'' +'.csv'
      ##領域,　使う成分（"Hemogrobin","Melanin","Shadow"）の設定
   ### checkROI.py
   ROIを確認（テスト）するためのもの
   ### making_separate_image.py
   色素成分分析画像を出力するためのもの

   ###  _funcSkinSeparation 
   色素成分分離を行った値を配列として返す
   ###  _funcSkinSeparation2
   色素成分分離を行った濃度空間のまま返す
   ### [2]  RGB_Pulse/program/_PredictPulse
      変更する箇所(main関数)
      INPUT_DIR =''
      OUTPUT_DIR =''
      subject
      (ほとんどの場合はsubjectを「１」で取得したcsvの名前に変更する)

### _bandpass.py
バンドパスフィルタをscipyライブラリを用いて作成したもの
### _detect_peak.py
バンドパスフィルタをscipyライブラリを用いて作成したもの
### _deTrending.py
デトレンド処理を行う
An Advanced Detrending Method With Application to HRV　を参照
### interploate_nan.py
データの補間を行う
### _detect_peak.py
バンドパスフィルタをscipyライブラリを用いて作成したもの




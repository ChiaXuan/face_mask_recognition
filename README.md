# face_mask_recognition
這是一個人臉辨識結合口罩辨識的程式。

## how it works
1. 訓練模型：
運用train.py檔案生成人臉辨識模型。
而口罩辨識則是用[Teachable Machine](https://teachablemachine.withgoogle.com/)來訓練戴口罩的模型(keras_model.h5)。
 
2. 辨識：
以recognition.py這個程式辨識畫面中的的人是誰以及他是否有配戴口罩。

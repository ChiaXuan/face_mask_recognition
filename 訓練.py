import cv2
import numpy as np

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # 載入人臉追蹤模型
recog = cv2.face.LBPHFaceRecognizer_create()      # 訓練人臉的模型
faces = []   # 儲存人臉位置大小的串列
ids = []     # 記錄該人臉 id 的串列
                            
cap = cv2.VideoCapture(0)                         
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, img = cap.read()                         
    if not ret:
        print("Cannot receive frame")
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    img_arr = np.array(gray,'uint8')
    print(img_arr)
    face = detector.detectMultiScale(gray)        # 擷取人臉區域
    for(x,y,w,h) in face:
        faces.append(img_arr[y:y+h,x:x+w])         # 記錄人臉部分的數值
        ids.append(1)                             # 記錄對應的 id
    
    cv2.imshow('trainer', img)                     
    if cv2.waitKey(100) == ord('q'):              # 按下 q 結束
        break

print('training...')                             
recog.train(faces,np.array(ids))                  # 開始訓練
recog.save('face.yml')                            # 訓練完成儲存為 face.yml
print('ok!')

cap.release()
cv2.destroyAllWindows()


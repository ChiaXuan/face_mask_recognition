import cv2
import tensorflow as tf
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()         
recognizer.read('face.yml')                               # 讀取人臉模型
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # 載入人臉追蹤模型
model = tf.keras.models.load_model('keras_model.h5', compile=False)  # 載入口罩辨識模型
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)          # 設定資料陣列 

def text(text):      # 建立顯示文字的函式
    global img       
    org = (20,50)     
    fontFace = cv2.FONT_HERSHEY_SIMPLEX  
    fontScale = 1                        
    color = (0,255,0)                
    thickness = 2                        
    cv2.putText(img, text, org, fontFace, fontScale, color, thickness) 

cap = cv2.VideoCapture(0)                                 
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
    img = cv2.resize(frame,(400, 250))              # 縮小尺寸，加快辨識效率
    
    pic = img[0:224, 80:304]
    image_arr = np.array(pic) 
    normalized_image_array = (image_arr.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array 
    prediction = model.predict(data) # 進行預測
    a,b,bg= prediction[0]    # 印出每個項目的數值資訊
    print(a,b,bg)
    
    name_dict = {
        '1':'Liao'
    }
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    faces = face_cascade.detectMultiScale(gray)
   
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)            # 標記人臉外框
        idnum,confidence = recognizer.predict(gray[y:y+h,x:x+w])  # 取出 id 號碼以及信心指數 
        print(confidence)
        if confidence < 75 and a>0.99:
            text("Good job, " + name_dict[str(idnum)] + "!")              
        if confidence < 75 and b>0.01:
            text(name_dict[str(idnum)] + ", no mask!")
        if confidence >=75 and a>0.99:
            text("Good job, stranger!")
        if confidence >= 75 and b>0.01:
            text('Stranger, no mask!')                                        
        
    cv2.imshow('recognizer', img)
    if cv2.waitKey(5) == ord('q'):   # 按下 q 鍵停止
        break    

cap.release()
cv2.destroyAllWindows()


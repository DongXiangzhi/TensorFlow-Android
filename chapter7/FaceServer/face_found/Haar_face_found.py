import cv2
img_path = './test.jpg' # 测试图片
# 模板匹配文件
cascade_path = './face_detection/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascade_path)
image = cv2.imread(img_path) # 读取图像
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# 人脸检测
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor = 1.1,
    minNeighbors = 4,
    minSize = (15,15)   # （20,20）-> 16
)
print(f'发现：{len(faces)} 张脸')
for(x,y,w,h) in faces:  # 画绿色框边界
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
cv2.putText(image, f'Haar Found Faces:{len(faces)}',(10,40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0),3)
cv2.imshow('Haar',image)
cv2.waitKey(0)






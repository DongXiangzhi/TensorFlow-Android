import cv2
import face_recognition
img_path = './test.jpg'  # 测试图片
image = face_recognition.load_image_file(img_path) #加载
# 检测人脸
face_locations = face_recognition.face_locations(image)
print(f'发现：{len(face_locations)} 张人脸')
image = cv2.imread(img_path) # 读取图片
# 对检测到的人脸加绿色边框
for (top,right,bottom,left) in face_locations:
    cv2.rectangle(image,(left,top),(right,bottom),(0,255,0),2)
cv2.putText(image, f'Dlib Found Faces:{len(face_locations)}',(10,40 ),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0),3)
cv2.imshow('Dlib',image)
cv2.waitKey(0)
import cv2
import dlib
img_path = './face_alignment.png'
image = cv2.imread(img_path)
detector = dlib.get_frontal_face_detector()
# landmark下载地址 http://dlib.net/files/
predictor = dlib.shape_predictor("./landmark/shape_predictor_68_face_landmarks.dat")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = detector(gray)
# 描绘landmark
for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    landmarks = predictor(gray, face)
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 4, (0, 255, 0), -1)
cv2.imshow('Landmark',image)
cv2.waitKey(0)
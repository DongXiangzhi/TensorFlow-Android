import cv2
from mtcnn import MTCNN
detector = MTCNN()
cap = cv2.VideoCapture(0)
# 逐帧检测
while True:
    __, frame = cap.read()
    # 用 MTCNN 检测人脸
    result = detector.detect_faces(frame)
    if result != []:
        for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']

            cv2.rectangle(frame,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0] + bounding_box[2],
                           bounding_box[1] + bounding_box[3]),
                          (0, 255, 0), 2)
            cv2.circle(frame, (keypoints['left_eye']), 2, (0, 255, 0), 3)
            cv2.circle(frame, (keypoints['right_eye']), 2, (0, 255, 0), 3)
            cv2.circle(frame, (keypoints['nose']), 2, (0, 255, 0), 3)
            cv2.circle(frame, (keypoints['mouth_left']), 2, (0, 255, 0), 3)
            cv2.circle(frame, (keypoints['mouth_right']), 2, (0, 255, 0), 3)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
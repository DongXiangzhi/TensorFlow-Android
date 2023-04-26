import cv2
from mtcnn import MTCNN  #pip install mtcnn
detector = MTCNN()
image = cv2.imread("test.jpg")
result = detector.detect_faces(image)
print(result[0])
bounding_box = result[0]['box']
keypoints = result[0]['keypoints']
for person in result:
    bounding_box = person['box']
    keypoints = person['keypoints']
    cv2.rectangle(image,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0]+bounding_box[2],
                   bounding_box[1] + bounding_box[3]),
                  (0,255,0), 2)
    cv2.circle(image,(keypoints['left_eye']), 2, (0,255,0), 2)
    cv2.circle(image,(keypoints['right_eye']), 2, (0,255,0), 2)
    cv2.circle(image,(keypoints['nose']), 2, (0,255,0), 3)
    cv2.circle(image,(keypoints['mouth_left']), 2,  (0,255,0), 2)
    cv2.circle(image,(keypoints['mouth_right']), 2, (0,255,0), 2)
cv2.putText(image, f'MTCNN Found Faces:{len(result)}',(10,40 ),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0),3)
cv2.imwrite("test_drawn.jpg", image)
cv2.namedWindow("MTCNN")
cv2.imshow("MTCNN",image)
cv2.waitKey(0)
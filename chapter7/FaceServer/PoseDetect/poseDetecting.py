import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp


mp_holistic = mp.solutions.holistic   # 姿态检测包 Holistic Model

actions = np.array(['stone','scissors','paper'])  # 动作标签
# 加载保存的最优模型
model = tf.keras.models.load_model('actions.h5')

sequence = []  # 存放帧序列
sentence = []  # 存放动作
threshold = 0.5

# 姿态检测
def mediapipe_detection(image,model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image,results
# 只采集左手的动作数据
def extract_left_hand_keypoints(results):
    # 对左手姿态数据处理
    lh = np.array([[res.x, res.y, res.z] \
                   for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)

    return lh


cap = cv2.VideoCapture(0)  # 打开摄像头
# 定义mediapipe模型
with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # 读取一帧
        ret, frame = cap.read()

        # 姿态检测
        image, results = mediapipe_detection(frame, holistic)

        # 绘制 landmarks
        # draw_styled_landmarks(image, results)

        # 捕获关键点的数据,这里只取左手动作数据，与采样数据保持一致
        keypoints = extract_left_hand_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # 取最近的30帧

        if len(sequence) == 30:  # 凑够30帧画面的数据
            pred = model.predict(np.expand_dims(sequence, axis=0))[0]  # 预测

            for num, prob in enumerate(pred):  # 可视化预测结果
                cv2.rectangle(image, (20, 20 + num * 40), (20 + int(prob * 100),
                             50 + num * 40), (0, 255, 255), -1)
                cv2.putText(image, actions[num] + ':' + str(prob), (20, 45 + num * 40),
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # 显示当前帧
        cv2.imshow('Current Frame', image)

        # ESC结束检测
        if cv2.waitKey(10) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
import cv2
import numpy as np
import os
import mediapipe as mp  # pip install mediapipe

mp_holistic = mp.solutions.holistic   # 姿态检测包 Holistic Model
mp_drawing = mp.solutions.drawing_utils   # 绘图包

# 姿态检测
def mediapipe_detection(image,model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image,results

# 绘制姿态关键点
def draw_styled_landmarks(image, results):
    # 绘制脸部轮廓
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,  \
                             mp_drawing.DrawingSpec(color=(80,110,10),thickness=1,circle_radius=1), \
                             mp_drawing.DrawingSpec(color=(80,256,121),thickness=1,circle_radius=1))
    # 绘制左手轮廓
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,  \
                             mp_drawing.DrawingSpec(color=(121,22,76),thickness=2,circle_radius=4), \
                             mp_drawing.DrawingSpec(color=(121,44,250),thickness=2,circle_radius=2))
    # 绘制右手轮廓
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,  \
                             mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=4), \
                             mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2))
    # 绘制躯体轮廓
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,  \
                             mp_drawing.DrawingSpec(color=(80,22,10),thickness=2,circle_radius=4), \
                             mp_drawing.DrawingSpec(color=(80,44,121),thickness=2,circle_radius=2))

# 采集全身动作数据，包括脸部、四肢、躯干
def extract_keypoints(results):
    # 对姿态数据的处理
    pose = np.array([[res.x, res.y, res.z, res.visibility] \
                     for res in results.pose_landmarks.landmark]).flatten() \
                     if results.pose_landmarks else np.zeros(33*4)
    # 对脸部姿态数据的处理
    face = np.array([[res.x, res.y, res.z] \
                     for res in results.face_landmarks.landmark]).flatten() \
                     if results.face_landmarks else np.zeros(468*3)
    # 对左手姿态数据处理
    lh = np.array([[res.x, res.y, res.z] \
                   for res in results.left_hand_landmarks.landmark]).flatten() \
                   if results.left_hand_landmarks else np.zeros(21*3)
    # 对右手姿态数据处理
    rh = np.array([[res.x, res.y, res.z] \
                   for res in results.right_hand_landmarks.landmark]).flatten() \
                   if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])  #返回动作堆叠合集

# 只采集左手的动作数据
def extract_left_hand_keypoints(results):
    # 对左手姿态数据处理
    lh = np.array([[res.x, res.y, res.z] \
                   for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)

    return lh

# 定义动作标签
actions = np.array(['stone','scissors','paper'])
datasets = os.path.join('datasets')  # 定义存放采集的位置
num_videos = 30   # 对每个动作，采集30个视频
frames_in_video = 30 # 每个动作，用30个Frame表示
# 创建数据集目录
for action in actions:
    for video in range(1,num_videos+1):
        try:
            os.makedirs(os.path.join(datasets,action,str(video)))
        except:
            pass

# 开始数据采样
end = False
cap = cv2.VideoCapture(0)  # 打开摄像头

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:  # 采集三种动作
        if end:
            break
        for video_no in range(num_videos):  # 每个动作采集多少个视频
            if end:
                break
            for frame_no in range(frames_in_video):  # 每个视频包含30帧
                # 读取一帧
                ret, frame = cap.read()

                # 姿态检测
                image, results = mediapipe_detection(frame, holistic)

                # 绘制检测的结果
                draw_styled_landmarks(image, results)

                # 显示当前帧
                if frame_no == 0:  # 显示当前序列的第1帧
                    cv2.putText(image, f'begin: {action}', (120, 200), \
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f'video no: {video_no + 1} / action: {action}', \
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('Current Frame', image)
                    cv2.waitKey(1000)  # 延迟1秒，准备下一个动作
                else:  # 显示当前序列的第2-30帧
                    cv2.putText(image, f'video no: {video_no + 1}/action:{action}', \
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('Current Frame', image)
                # 提取和保存关键数据
                keypoints = extract_keypoints(results)
                # keypoints = extract_left_hand_keypoints(results) # 只提取左手动作
                saved_path = os.path.join(datasets, action, str(video_no + 1), str(frame_no + 1))
                np.save(saved_path, keypoints) # 保存为npy数据文件
                # 保存采集的图像，测试和观察用
                # plt.imsave(saved_path+'.jpg',cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # ESC键退出视频采集
                if cv2.waitKey(10) & 0xFF == 27:
                    end = True
                    break

    cap.release()
    cv2.destroyAllWindows()
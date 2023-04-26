import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
# 加载并划分数据集
actions = np.array(['stone','scissors','paper'])  # 动作标签
datasets = os.path.join('datasets')  # 存放采集的所有数据
num_videos = 30   # 对每个动作，采集30个视频
frames_in_video = 30 # 每个动作，用30个Frame表示
frames, labels = [], []  # 所有的序列，及其标签
label_map = {label : num for num , label in enumerate(actions)}
for action in actions:
    for video_no in range(1, num_videos+1):
        action_frames = []
        for frame_no in range(1, frames_in_video+1):
            keypoints = np.load(os.path.join(datasets,action,str(video_no),f"{str(frame_no)}.npy"))
            action_frames.append(keypoints)
        frames.append(action_frames)
        labels.append(label_map[action])
X = np.array(frames)  # 特征矩阵
y = to_categorical(labels).astype(int)  # 标签
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2022)

# 用LSTM解析动作序列，定义模型 PoseModel
model = Sequential(name='PoseModel')
model.add(LSTM(256, return_sequences=True, activation='relu', input_shape=(30,63)))
model.add(LSTM(256, return_sequences=True, activation='relu'))
model.add(LSTM(256, return_sequences=False, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
# 模型编译，优化算法、损失函数、评价方法
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 模型训练日志目录及回调函数
log_dir = os.path.join('Logs')
tensorboard = TensorBoard(log_dir=log_dir)
# 定义保存最优模型的策略
best_model = tf.keras.callbacks.ModelCheckpoint(
    'actions.h5',
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    save_freq='epoch'
)
# 模型提前终止训练的策略
earlyStop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    min_delta=0,
    patience=100,
    verbose=0,
    mode='max',
    restore_best_weights=True
)
# 回调函数集合
callbacks = [best_model, tensorboard, earlyStop]
# 模型开始训练
# model.fit(X_train, y_train, epochs=1000,
#           validation_data =(X_test, y_test),
#           callbacks=callbacks)


# 绘制混淆矩阵
from sklearn.metrics import confusion_matrix, accuracy_score
import itertools
import matplotlib.pyplot as plt

# 加载保存的最优模型
model = tf.keras.models.load_model('actions.h5')
yhat = model.predict(X_test)  # 在验证集上做预测
print('预测结果：',actions[np.argmax(yhat[4])]) # 打印预测结果
print('真实标签：',actions[np.argmax(y_test[4])])  # 打印标签

# 得到真实标签和预测的标签
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
# 根据预测结果生成混淆矩阵
cm = confusion_matrix(ytrue, yhat)
# 用图形化方式绘制混淆矩阵
plt.figure(figsize=(4,3))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks(range(3))
plt.yticks(range(3))
plt.title("Confusion Matrix")
thresh = cm.max() / 2
for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
    plt.text(j,i,format(cm[i,j],'d'),
            horizontalalignment='center',
            color = 'white' if cm[i,j]>thresh else 'black')
plt.xlabel('Predicted Labbel')
plt.ylabel('True Label')
plt.show()
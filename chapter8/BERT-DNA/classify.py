# ===========================================
# bert_feature.py
# 功能：用 BERT 模型提取DNA序列特征，特征和标签存入hdf5文件
# 设计： 董相志
# 日期： 2022.2.20
# ===========================================

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


model = tf.keras.Sequential()  # 定义模型
# 采用DenseNet201的预训练模型作为分类基础模型
base_model = tf.keras.applications.densenet.DenseNet121(include_top=False,
                                                   weights='imagenet',
                                                   input_shape=(200, 256, 3))
base_model.trainable = True  # 微调训练模式
model.add(base_model)  # 以 DenseNet201 作为基础模型
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

# 模型编译
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# 加载训练集
f = h5py.File('./dna_train.hdf5','r')
X_train = f['X_train'][...]
y_train = f['y_train'][...]
f.close()
X_train = X_train.reshape((len(X_train), 200, 256,3))
y_train = np.squeeze(y_train)

# 划分训练集为两部分，训练样本占比90%，验证样本占比10%
X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                  y_train,
                                                  shuffle=True,
                                                  test_size=0.1,
                                                  random_state=2022)
BATCH_SIZE = 16
EPOCHS = 20
history = model.fit(X_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(X_val,y_val)).history
# 绘制训练曲线
def plot_learning_curves(history,label):
    plt.figure(figsize=(8,6))
    x = range(1,len(history[label])+1)
    plt.plot(x, history[label], label='train_'+label)
    plt.plot(x, history['val_'+label],label='val_'+label)
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.legend()
    plt.savefig(label+'.png')
    plt.show()
# 绘制准确率曲线
plot_learning_curves(history, 'accuracy')
# 绘制损失函数曲线
plot_learning_curves(history, 'loss')
# 在测试集上评估
# 加载测试集
f = h5py.File('./dna_test.hdf5','r')
X_test = f['X_test'][...]
y_test = f['y_test'][...]
f.close()
X_test = X_test.reshape((len(X_test), 200, 256, 3))
y_test = np.squeeze(y_test)
score = model.evaluate(X_test, y_test)  # 评估
print(f'测试集上的准确率：{score[1]}')


# 功能：MobileNetV3鸟类模型预测
# 设计：董相志
# 日期：2022.1.16
# ====================================================

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from mobilenet_v3 import HardSwish, HardSigmoid
from tensorflow.keras.preprocessing import image_dataset_from_directory

# 图像归一化
def normalize_image(image, label):
    return tf.cast(image, tf.float32) / 255., label


def main():
    img_height = 224
    img_width = 224

    # 从数据集（测试集、验证集、训练集）随机选择四幅图片测试
    # img_path = "dataset/birds/test/INDIAN ROLLER/1.jpg"
    # img_path = "dataset/birds/train/ALBATROSS/001.jpg"
    # img_path = "dataset/birds/valid/COUCHS KINGBIRD/5.jpg"
    # img_path = "dataset/birds/test/WHITE TAILED TROPIC/5.jpg"
    img_path = "dataset/birds/test/AMERICAN GOLDFINCH/1.jpg"
    img = Image.open(img_path)
    img = img.resize((img_width, img_height))  # 重新缩放尺寸
    plt.imshow(img)
    img = np.array(img) / 255.
    # 扩展特征矩阵维度，满足模型输入需要(bath,height,weight,channel)
    img = (np.expand_dims(img, 0))

    # 读取标签列表
    class_dict = pd.read_csv('dataset/birds/class_dict.csv')
    classes = class_dict['class'].tolist()
    pd.DataFrame(class_dict['class']).to_csv('labels.txt',
                                             index=False, header=None)

    # 加载已经训练完成的鸟类识别模型
    model_path = './saved_model/birds_model.h5'
    model = tf.keras.models.load_model(model_path,
                                       custom_objects={'HardSwish': HardSwish,
                                                       'HardSigmoid': HardSigmoid})

    result = model.predict(img)[0]  # 预测
    class_index = np.argmax(result) # 最大概率对应的索引
    # 预测结果
    title = "Label: {}   prob: {:.3}".format(classes[class_index],
                                             result[class_index])
    plt.title(title)
    plt.show()  # 显示图片和预测结果
    for i in range(len(result)):  # 控制台观察各类别预测值
        print("Label:{:10}  prob:{:.3}".format(classes[i],result[i]))

    # 评估模型在整个测试集和验证集上的表现
    test_dir = './dataset/birds/test'
    test_ds = image_dataset_from_directory(test_dir,
                                           image_size=(img_height,img_width),
                                           label_mode='categorical')
    test_ds = test_ds.map(normalize_image)  # 测试集归一化
    valid_dir = './dataset/birds/valid'
    valid_ds = image_dataset_from_directory(valid_dir,
                                           image_size=(img_height, img_width),
                                           label_mode='categorical')
    valid_ds = valid_ds.map(normalize_image)  # 验证集归一化
    model.evaluate(valid_ds)  # 模型在验证集上的准确率
    model.evaluate(test_ds)  # 模型在测试集上的准确率

    # 将当前模型保存为 tf 格式，便于后续将其转换为TFLite格式。
    saved_model_path = './saved_model/mobilenetv3_birds_model'
    model.save(saved_model_path)

if __name__ == '__main__':
    main()

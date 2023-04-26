# 功能：EfficientNetV2模型预测
# 参考论文原作者发布的源码和GitHub作者WZMIAOMIAO发布的源码改编
# https://github.com/google/automl/tree/master/efficientnetv2
# https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/tensorflow_classification/Test11_efficientnetV2
# ==============================================================================

import os
import json
import glob
import numpy as np

from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from model import efficientnetv2_s as create_model


def main():
    num_classes = 5

    img_size = {"s": 384,
                "m": 480,
                "l": 480}
    num_model = "s"
    im_height = im_width = img_size[num_model]

    # 测试图片
    img_path = "test_pic/dandelion.jpg"
    assert os.path.exists(img_path), "文件: '{}'不存在！".format(img_path)
    img = Image.open(img_path)
    # 图片缩放
    img = img.resize((im_width, im_height))
    plt.imshow(img)  # 显示图片

    # 转换数据类型， numpy 矩阵
    img = np.array(img).astype(np.float32)

    # 归一化 [-1,1]
    img = (img / 255. - 0.5) / 0.5

    # 扩展维度，[batch,height,width,channel]
    img = (np.expand_dims(img, 0))

    # 读取类别标签
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "文件: '{}' 不存在！".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # 创建模型
    model = create_model(num_classes=num_classes)
    # 加载权重
    weights_path = './save_weights/my_efficientnetv2.ckpt'
    assert len(glob.glob(weights_path+"*")), "找不到：{}".format(weights_path)
    model.load_weights(weights_path)
    # 模型预测，用模型对图片做推断
    result = np.squeeze(model.predict(img))
    result = tf.keras.layers.Softmax()(result) # 输出各类别概率
    predict_class = np.argmax(result)  # 最大概率对应的索引

    print_res = "class: {}   prob: {:.3}"\
        .format(class_indict[str(predict_class)],result[predict_class])
    plt.title(print_res)  # 图片标题为预测结果
    plt.show()
    for i in range(len(result)): # 在控制台返回所有类别预测结果
        print("class: {:10}   prob: {:.3}"
              .format(class_indict[str(i)],result[i].numpy()))


if __name__ == '__main__':
    main()

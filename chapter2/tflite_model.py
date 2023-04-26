# 功能：MobileNetV3模型转为TFLite格式，并与原始模型比较
# 设计：董相志
# 日期：2022.1.16
# ====================================================
import tensorflow as tf
import tensorflow.lite as lite
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from mobilenet_v3 import HardSwish,HardSigmoid
from tensorflow.keras.preprocessing import image_dataset_from_directory
# 模型转换
# saved_model_dir ='./saved_model/mobilenetv3_birds_model'   # 训练好的模型
# converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)  # 转换
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]  # 量化优化
# tflite_model = converter.convert()

saved_tflite_model_dir = './saved_model/android_birds_model.tflite'
# with open(saved_tflite_model_dir,'wb') as f: # 保存tflite模型
#     f.write(tflite_model)


img_height = 224
img_width = 224

# 从数据集（测试集、验证集、训练集）随机选择四幅图片测试
# img_path = "dataset/birds/test/INDIAN ROLLER/1.jpg"
img_path = "dataset/birds/test/AMERICAN GOLDFINCH/1.jpg"
# img_path = "dataset/birds/train/ALBATROSS/001.jpg"
# img_path = "dataset/birds/valid/COUCHS KINGBIRD/5.jpg"
# img_path = "dataset/birds/test/WHITE TAILED TROPIC/5.jpg"

img = Image.open(img_path)
img = img.resize((img_width, img_height))  # 重新缩放尺寸
plt.imshow(img)
img = np.array(img).astype('float32')
img = img / 255.0
# 扩展特征矩阵维度，满足模型输入需要(bath,height,weight,channel)
img = np.expand_dims(img, 0)

# 读取标签列表
class_dict = pd.read_csv('dataset/birds/class_dict.csv')
classes = class_dict['class'].tolist()

# 加载TFLite鸟类识别模型,创建TFLite模型解释器
interpreter = tf.lite.Interpreter(model_path=saved_tflite_model_dir)

#  TFLite 模型推理
def lite_model(images):
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], images)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])


probs_lite = lite_model(img)[0]  # 对图片img推理
class_index = np.argmax(probs_lite) # 最大概率对应的索引
# 预测结果
title = "Label: {}   prob: {:.3}".format(classes[class_index],
                                         probs_lite[class_index])
plt.title(title)
plt.show()  # 显示图片和预测结果

# 用测试集评估TFLite模型与转换前的原始模型，
# 加载已经训练完成的鸟类识别模型
model_path = './saved_model/birds_model.h5'
model = tf.keras.models.load_model(model_path,
                                   custom_objects={'HardSwish': HardSwish,
                                                   'HardSigmoid': HardSigmoid})

test_dir = './dataset/birds/test'
test_ds = image_dataset_from_directory(test_dir,
                                       image_size=(img_height, img_width),
                                       label_mode='categorical')

num_eval_examples = 100
eval_dataset = test_ds.unbatch()  # TFLite需要将batch_size设为1
count = 0
count_lite_tf_agree = 0  # 记录两种模型预测结果一致的数量
count_lite_correct = 0  # 记录TFLite模型预测正确的数量

print('正在对TFLite与原始模型在随机抽测的数据集上做评估,稍后片刻...')
for image, label in eval_dataset:  # 遍历测试数据集
    probs_lite = lite_model(image[None, ...]/255.)[0]  # TFLite预测
    probs_tf = model(image[None, ...]/255.).numpy()[0]  # 原始模型预测
    y_lite = np.argmax(probs_lite)  # 最大索引
    y_tf = np.argmax(probs_tf)
    y_true = np.argmax(label)  # 正确标签
    count +=1
    if y_lite == y_tf: count_lite_tf_agree += 1  # 统计结果一致的数量
    if y_lite == y_true: count_lite_correct += 1 # 统计正确数量
    if count >= num_eval_examples: break

print(f"TFLite模型与转换前的原始模型相比，随机抽测的 {count} 个样本中，"
      f"有 {count_lite_tf_agree} 个预测结果保持一致，一致性达到："
      f"{100.0 * count_lite_tf_agree / count}%")
print(f"TFLite 模型在随机抽测的 {count} 个样本上的正确预测为："
      f"{count_lite_correct}个，正确率：{100.0 * count_lite_correct / count}%")
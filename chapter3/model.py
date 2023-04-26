# ===========================================
# model.py
# 功能：EfficientDe-Lite版美食场景检测，基于tflite-model-maker的迁移学习
# 设计： 董相志
# 日期： 2022.3.3
# ===========================================
import json
from absl import logging
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
logging.set_verbosity(logging.ERROR)

spec = model_spec.get('efficientdet_lite0')  # 指定模型

print('数据集划分需要读取14611幅图像，可能花费几分钟时间。请耐心等待！')
train_data, validation_data, test_data = object_detector.DataLoader\
    .from_csv('./dataset100/datasets.csv')
print('开始模型训练...')
# 训练模型，指定训练参数
model = object_detector.create(train_data,
                               model_spec=spec,
                               epochs=30,
                               batch_size=16,
                               train_whole_model=True,
                               validation_data=validation_data)

# 将训练好的模型导出为TFLite model并保存到当前工作目录下。默认采用整数量化方法
print('正在采用默认优化方法，保存TFLite模型...')
model.export(export_dir='.')  # 保存TFLite模型
model.summary()

# 保存与模型输出一致的标签列表
classes = ['???'] * model.model_spec.config.num_classes
label_map = model.model_spec.config.label_map
for label_id, label_name in label_map.as_dict().items():
    classes[label_id-1] = label_name
print(classes)
with open('labels.txt', 'w') as f:  # 模型标签保存到文件
    for i in range(len(classes)):
        for label in classes:
            f.write(label+"\r")
# 在测试集上评测训练好的模型
dict1 = {}
print('开始在测试集上对电脑版模型评估...')
dict1 = model.evaluate(test_data, batch_size = 16)
print(f'电脑版模型在测试集上评估结果：\n {dict1}')

# 加载TFLite格式的模型，在测试集上做评估
dict2 = {}
print('开始在测试集上对优化后的TFLite模型评估....')
dict2 = model.evaluate_tflite('model.tflite', test_data)
print(f'优化后的TFLite模型在测试集上评估结果： \n {dict2}')

# 保存模型的评估结果
for key in dict1:
    dict1[key] = str(dict1[key])
    print(f'{key}: {dict1[key]}')
with open('dict1.txt','w') as f :
    f.write(json.dumps(dict1))

# 保存优化后的TFLite模型在测试集上的评估结果
print('真实版的TFLite模型测试结果...')
for key in dict2:
    dict2[key] = str(dict2[key])
    print(f'{key}: {dict2[key]}')
with open('dict2.txt','w') as f :
    f.write(json.dumps(dict2))
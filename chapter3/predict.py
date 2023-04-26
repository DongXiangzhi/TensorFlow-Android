# ===========================================
# predict.py
# 功能：EfficientDe-Lite美食场景检测模型测试
# 设计： 董相志
# 日期： 2022.3.6
# ===========================================


import cv2
import tensorflow as tf
from PIL import Image
import numpy as np


model_path = 'model.tflite'  # 预训练模型

with open('labels.txt','r') as f:  # 读取模型标签文件
    classes = f.readlines()
for i in range(len(classes)):  # 取出标签中的换行符
    classes[i] = classes[i].replace('\n','')


# 图像预处理
def preprocess_image(image_path, input_size):
    img = tf.io.read_file(image_path)  # 读取指定图像
    img = tf.io.decode_image(img, channels=3)  # 解码
    img = tf.image.convert_image_dtype(img, tf.uint8)  # 数据类型
    original_image = img  # 原始图像
    resized_img = tf.image.resize(img, input_size)  # 图像缩放
    resized_img = resized_img[tf.newaxis, :]  # 增加维度，表示样本数量
    resized_img = tf.cast(resized_img, dtype=tf.uint8)  # 数据类型
    return resized_img, original_image  # 裁剪后的图像与原始图像


def detect_objects(interpreter, image, threshold):
    """
    用指定的模型和置信度阈值，对指定的图像检测
    :param interpreter: 推理模型
    :param image: 待检测图像
    :param threshold: 置信度阈值
    :return: 返回检测结果（字典列表）
    """
    # 推理模型
    signature_fn = interpreter.get_signature_runner()

    # 对指定图像做目标检测
    output = signature_fn(images=image)

    # 解析检测结果
    count = int(np.squeeze(output['output_0']))  # 检测到的目标数量
    scores = np.squeeze(output['output_1'])  # 置信度
    class_curr = np.squeeze(output['output_2'])  # 类别
    boxes = np.squeeze(output['output_3'])  # Bounding box坐标

    results = []
    for i in range(count):   # 所有目标组织为列表
        if scores[i] >= threshold:  # 只返回超过阈值的目标
            result = {  # 以字典格式组织单个检测结果
              'bounding_box': boxes[i],
              'class_id': class_curr[i],
              'score': scores[i]
            }
            results.append(result)
    return results  # 返回检测结果（字典列表）


def run_odt_and_draw_results(image_path, interpreter, threshold=0.5):
    """
    用指定模型在指定图片上根据阈值做目标检测并绘制检测结果
    :param image_path: 待检测图像
    :param interpreter: 推理模型
    :param threshold: 智信度阈值
    :return: 绘制Bounding box、类别和置信度的图像数组
    """
    # 根据模型获得输入维度
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    # 加载图像并做预处理
    preprocessed_image, original_image = preprocess_image(
        image_path,
        (input_height, input_width)
      )

    # 对图像做目标检测
    results = detect_objects(interpreter,
                             preprocessed_image,
                             threshold=threshold)

    # 在图像上绘制检测结果（Bounding box，类别，置信度）
    original_image_np = original_image.numpy().astype(np.uint8)
    for obj in results:
        # 根据原始图像尺寸（高度和宽度），将Bounding box的坐标调整为整数，
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * original_image_np.shape[1])
        xmax = int(xmax * original_image_np.shape[1])
        ymin = int(ymin * original_image_np.shape[0])
        ymax = int(ymax * original_image_np.shape[0])

        # 当前类别的ID
        class_id = int(obj['class_id'])

        # 用指定颜色绘制Bounding box
        color = [0,255,0]
        cv2.rectangle(original_image_np,
                      (xmin, ymin),
                      (xmax, ymax),
                      color, 1)
        # 调整类别标签的纵向坐标，保持可见
        y = ymin - 5 if ymin - 5 > 15 else ymin + 20
        # 类别标签和置信度显示为字符串
        label = "{}: {:.0f}%".format(classes[class_id],
                                     obj['score'] * 100)
        color = [255,255,0]  # 标签文本颜色
        cv2.putText(original_image_np, label, (xmin+5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # 绘制标签
    # 返回绘制结果的图像
    original_uint8 = original_image_np.astype(np.uint8)
    return original_uint8


# 随机选择图像进行测试
TEMP_FILE = './dataset100/25.jpg'
# TEMP_FILE = './dataset100/11156.jpg'
DETECTION_THRESHOLD = 0.13   # 置信度阈值，可以调整

im = Image.open(TEMP_FILE)  # 打开图像
im.thumbnail((512, 512), Image.ANTIALIAS)  # 缩放
im.save(TEMP_FILE)  # 保存缩放后的图像

# 加载TFLite推理模型
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 目标检测并绘制检测结果
detection_result_image = run_odt_and_draw_results(
    TEMP_FILE,
    interpreter,
    threshold=DETECTION_THRESHOLD
)

# 显示检测结果
Image.fromarray(detection_result_image).show()
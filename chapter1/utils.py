# 功能：数据集预处理
# 参考论文原作者发布的源码和GitHub作者WZMIAOMIAO发布的源码改编
# https://github.com/google/automl/tree/master/efficientnetv2
# https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/tensorflow_classification/Test11_efficientnetV2
# ==============================================================================

import os
import json
import random

import tensorflow as tf


# 数据集读取与划分（只在路径和标签层次划分，不涉及图像数据）
def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(2022)  # 保证随机划分结果一致
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，用文件夹名称作为类别名称
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 列表排序
    flower_class.sort()
    # 生成索引、类别名称组成的键值对，形成 json 字符串保存到文件中
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 训练集的所有图片路径
    train_images_label = []  # 训练集图片对应的标签
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 验证集图片对应的标签
    every_class_num = []  # 每个类别的样本总数
    supported = [".jpg", ".JPG", ".jpeg", ".JPEG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的数值
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机抽样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 验证集图片及标签归入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则归入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("共包含 {} 幅图片。\n{} 幅图片划入训练集, {} 幅图片划入验证集"
          .format(sum(every_class_num),
                  len(train_images_path),
                  len(val_images_path)))

    return train_images_path, train_images_label, \
           val_images_path, val_images_label


# 读取划分数据集，并生成训练集和验证集的迭代器
def generate_ds(data_root: str,  # 数据集根目录
                train_im_height: int = None,  # 训练图片的高
                train_im_width: int = None,  # 训练图片的宽
                val_im_height: int = None,  # 验证图片的高
                val_im_width: int = None,  # 验证图片的宽
                batch_size: int = 8,
                val_rate: float = 0.1,  # 验证集占比
                cache_data: bool = False  # 是否缓存数据
                ):
    assert train_im_height is not None
    assert train_im_width is not None
    if val_im_width is None:
        val_im_width = train_im_width
    if val_im_height is None:
        val_im_height = train_im_height

    train_img_path, train_img_label, val_img_path, val_img_label = \
        read_split_data(data_root, val_rate=val_rate)  # 划分
    AUTOTUNE = tf.data.experimental.AUTOTUNE  # 数据集并发性调为自动

    def process_train_info(img_path, label): # 训练集预处理
        image = tf.io.read_file(img_path)  # 读图片
        image = tf.image.decode_jpeg(image, channels=3)  # 解码
        image = tf.cast(image, tf.float32)  # 数据类型转换
        image = tf.image.resize_with_crop_or_pad(image,   # 裁剪
                                                 train_im_height,
                                                 train_im_width)
        image = tf.image.random_flip_left_right(image)  # 水平翻转
        image = (image / 255. - 0.5) / 0.5   # 归一化 [-1,1]
        return image, label

    def process_val_info(img_path, label):  # 验证集预处理
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_with_crop_or_pad(image,  # 裁剪
                                                 val_im_height,
                                                 val_im_width)
        image = (image / 255. - 0.5) / 0.5  # 归一化[-1,1]
        return image, label

    # 配置数据集性能
    def configure_for_performance(ds,
                                  shuffle_size: int,
                                  shuffle: bool = False,
                                  cache: bool = False):
        if cache:
            ds = ds.cache()  # 读取数据后缓存至内存
        if shuffle:
            ds = ds.shuffle(buffer_size=shuffle_size)  # 打乱顺序
        ds = ds.batch(batch_size)  # 指定 batch size
        ds = ds.prefetch(buffer_size=AUTOTUNE)  # 训练时提前准备下一个step的数据
        return ds

    # 此时算是真正的训练集，数据形式为样本的特征矩阵
    train_ds = tf.data.Dataset.from_tensor_slices((
        tf.constant(train_img_path),tf.constant(train_img_label)))
    total_train = len(train_img_path)

    # Use Dataset.map to create a dataset of image, label pairs
    train_ds = train_ds.map(process_train_info, num_parallel_calls=AUTOTUNE)
    train_ds = configure_for_performance(train_ds,
                                         total_train,
                                         shuffle=True,
                                         cache=cache_data)

    val_ds = tf.data.Dataset.from_tensor_slices((tf.constant(val_img_path),
                                                 tf.constant(val_img_label)))
    total_val = len(val_img_path)
    # Use Dataset.map to create a dataset of image, label pairs
    val_ds = val_ds.map(process_val_info, num_parallel_calls=AUTOTUNE)
    val_ds = configure_for_performance(val_ds, total_val, cache=False)

    return train_ds, val_ds

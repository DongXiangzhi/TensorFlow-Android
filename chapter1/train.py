# 功能：EfficientNetV2模型训练
# 参考论文原作者发布的源码和GitHub作者WZMIAOMIAO发布的源码改编
# https://github.com/google/automl/tree/master/efficientnetv2
# https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/tensorflow_classification/Test11_efficientnetV2
# ==============================================================================

import os
import sys
import math
import datetime
import tensorflow as tf
from tqdm import tqdm
from model import efficientnetv2_s as create_model
from utils import generate_ds

def main():
    data_root = "dataset/flower_photos"  # 数据集的根目录

    if not os.path.exists("./save_weights"):  # 权重保存路径
        os.makedirs("./save_weights")

    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "s"

    batch_size = 8
    epochs = 20
    num_classes = 5
    freeze_layers = True  # 只训练模型的最后一个Stage，即顶层
    initial_lr = 0.01  # 学习率初始值

    # 日志文件目录，保存训练过程中产生的数据，如accuracy，可以用TensorBoard查看
    log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
    val_writer = tf.summary.create_file_writer(os.path.join(log_dir, "val"))

    # 数据集处理，包括随机划分、数据增强等
    train_ds, val_ds = generate_ds(data_root,
                                   train_im_height=img_size[num_model][0],
                                   train_im_width=img_size[num_model][0],
                                   val_im_height=img_size[num_model][1],
                                   val_im_width=img_size[num_model][1],
                                   batch_size=batch_size)

    # 创建模型结构
    model = create_model(num_classes=num_classes)
    model.build((1, img_size[num_model][0], img_size[num_model][0], 3))

    # 可以下载 WZMIAOMIAO放在百度网盘上的预训练权重
    # 链接: https://pan.baidu.com/s/1Pr-pO5sQVySPQnBY8pQH7w  密码: f6hi
    # 或者也可以自行转换，请参见本节的视频讲解
    pre_weights_path = './efficientnetv2-s.h5'
    assert os.path.exists(pre_weights_path), "cannot find {}".format(pre_weights_path)
    model.load_weights(pre_weights_path, by_name=True, skip_mismatch=True)

    # 模型中需要固定的层
    if freeze_layers:
        unfreeze_layers = "head"
        for layer in model.layers:
            if unfreeze_layers not in layer.name:
                layer.trainable = False  # 训练期间参数固定不变
            else:
                print("training {}".format(layer.name))
    # 观察模型结构
    model.summary(input_shape=(img_size[num_model][0], img_size[num_model][0], 3))

    # 根据epoch定义学习率调度策略
    def scheduler(now_epoch):
        end_lr_rate = 0.01  # end_lr = initial_lr * end_lr_rate
        rate = ((1 + math.cos(now_epoch * math.pi / epochs)) / 2) \
               * (1 - end_lr_rate) + end_lr_rate  # cosine
        new_lr = rate * initial_lr

        # 保存学习率衰减过程到训练日志文件中，后面可用TensorBoard查看
        with train_writer.as_default():
            tf.summary.scalar('learning rate', data=new_lr, step=epoch)

        return new_lr

    # 定义损失函数分类交叉熵损失、优化算法SGD、评价标准准确率
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=0.9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    @tf.function
    def train_step(train_images, train_labels):  # 单步训练逻辑
        with tf.GradientTape() as tape:
            output = model(train_images, training=True)  # 正向传播
            loss = loss_object(train_labels, output)  # 计算损失
        # 反向传播，计算梯度
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)  # 训练集的单步损失均值
        train_accuracy(train_labels, output)  # 训练集的单步准确率均值

    @tf.function
    def val_step(val_images, val_labels):  # 单步验证逻辑
        output = model(val_images, training=False)  # 正向传播
        loss = loss_object(val_labels, output)  # 计算损失

        val_loss(loss)  # 验证集的单步损失均值
        val_accuracy(val_labels, output)  # 验证集的单步准确率均值

    best_val_acc = 0.
    for epoch in range(epochs):  # 按照 epoch 迭代训练
        train_loss.reset_states()  # 清空历史数据
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        # 完成一代训练
        train_bar = tqdm(train_ds, file=sys.stdout)
        for images, labels in train_bar:
            train_step(images, labels)

            # 每一步训练的结果，包括损失和准确率
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}"\
                .format(epoch + 1,epochs,
                        train_loss.result(), train_accuracy.result())

        # 调度学习率
        optimizer.learning_rate = scheduler(epoch)

        # 完成一代验证
        val_bar = tqdm(val_ds, file=sys.stdout)
        for images, labels in val_bar:
            val_step(images, labels)

            # 每一步的验证结果
            val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}"\
                .format(epoch + 1,epochs,
                        val_loss.result(),val_accuracy.result())
        # 保存训练结果到日志文件
        with train_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), epoch)
            tf.summary.scalar("accuracy", train_accuracy.result(), epoch)

        # 保存验证结果到日志文件
        with val_writer.as_default():
            tf.summary.scalar("loss", val_loss.result(), epoch)
            tf.summary.scalar("accuracy", val_accuracy.result(), epoch)

        # 保留最佳模型
        if val_accuracy.result() > best_val_acc:
            best_val_acc = val_accuracy.result()
            save_name = "./save_weights/my_efficientnetv2.ckpt"
            model.save_weights(save_name, save_format="tf")


if __name__ == '__main__':
    main()

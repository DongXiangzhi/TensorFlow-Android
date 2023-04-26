# 功能：MobileNetV3模型训练
# 设计：董相志
# 日期：2022.1.16
# ====================================================

import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory
from mobilenet_v3 import mobilenet_v3_large

# 图像归一化
def normalize_image(image, label):
    return tf.cast(image, tf.float32) / 255., label


def main():
    data_root = "dataset/birds"  # 数据集根目录

    img_height = 224
    img_width = 224
    epochs = 20
    num_classes = 325
    freeze_layer = False  # 控制模型的迁移学习模式

    # 加载数据集，返回训练集和验证集
    train_dir = os.path.join(data_root, 'train')
    train_ds = image_dataset_from_directory(train_dir,
                                           image_size=(img_height, img_width),
                                           label_mode='categorical')

    train_ds = train_ds.map(normalize_image)  # 训练集归一化

    valid_dir = os.path.join(data_root, 'valid')
    valid_ds = image_dataset_from_directory(valid_dir,
                                            image_size=(img_height, img_width),
                                            label_mode='categorical')
    valid_ds = valid_ds.map(normalize_image)  # 验证集归一化

    # 从测试集抽取样本观察
    test_dir = os.path.join(data_root, 'test')
    test_ds = image_dataset_from_directory(test_dir, label_mode='int')
    class_names = test_ds.class_names
    plt.figure(figsize=(12, 12))
    for images, labels in test_ds.take(1):
        for i in range(9): # 从当前batch抽取9幅图片显示
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()

    # 创建模型实例
    model = mobilenet_v3_large(input_shape=(img_height, img_width, 3),
                               num_classes=num_classes,
                               include_top=True)
    # 加载权重
    pre_weights_path = './weights_mobilenet_v3_large_224_1.0_float.h5'
    assert os.path.exists(pre_weights_path), "cannot find {}".format(pre_weights_path)
    model.load_weights(pre_weights_path, by_name=True, skip_mismatch=True)

    if freeze_layer is True:
        # 冻结层参数，只训练最后两层
        for layer in model.layers:
            if layer.name not in ["Conv_2", "Logits/Conv2d_1c_1x1"]:
                layer.trainable = False
            else:
                print("training: " + layer.name)

    model.summary()
    model.compile(optimizer='adam',loss="categorical_crossentropy",
                  metrics=['accuracy'])
    best_model = tf.keras.callbacks.ModelCheckpoint(  # 最优模型保存策略
        './saved_model/birds_model.h5',
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto')
    # 可视化训练过程
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs')
    # 如果连续 10 个 Epoch 损失函数曲线不下降，模型则训练提前终止
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 min_delta=0,
                                                 patience=10,
                                                 verbose=1,
                                                 restore_best_weights=True)

    # 开始训练过程
    model.fit(
        train_ds,  # 训练集
        epochs=epochs,
        validation_data=valid_ds,  # 验证集
        callbacks=[best_model,tensorboard,earlyStop]  # 回调函数
    )


if __name__ == '__main__':
    main()

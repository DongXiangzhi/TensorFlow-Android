# 功能：EfficientNetV2模型实现
# 参考论文原作者发布的源码和GitHub作者WZMIAOMIAO发布的源码改编
# https://github.com/google/automl/tree/master/efficientnetv2
# https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/tensorflow_classification/Test11_efficientnetV2
# ==============================================================================

import itertools
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input


# 卷积层参数初始化
def conv_kernel_initializer(shape, dtype=None):
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random.normal(
        shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


# 全连接层参数初始化
def dense_kernel_initializer(shape, dtype=None):
    init_range = 1.0 / np.sqrt(shape[1])
    return tf.random.uniform(shape, -init_range, init_range, dtype=dtype)


# SE模块层
class SE(layers.Layer):
    def __init__(self,
                 se_filters,  # 第一层节点个数
                 output_filters,  # 第二层节点个数
                 name=None):
        super().__init__(name=name)
        # SE包含两层，这是第一层，用1x1卷积定义
        self.se_reduce = layers.Conv2D(
            se_filters,
            kernel_size=1,
            strides=1,
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            activation='swish',
            use_bias=True,
            name='conv2d')

        self.se_expand = layers.Conv2D(
            output_filters,
            kernel_size=1,
            strides=1,
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            activation='sigmoid',
            use_bias=True,
            name='conv2d_1')

    # SE模块的逻辑实现
    def call(self, inputs, **kwargs):
        # Tensor: [N, H, W, C] -> [N, 1, 1, C]，全局平均池化
        se_tensor = layers.GlobalAveragePooling2D(keepdims=True)(inputs)
        se_tensor = self.se_reduce(se_tensor)  # 第一个卷积层
        se_tensor = self.se_expand(se_tensor)  # 第二个卷积层
        return se_tensor * inputs  # 相乘，得到SE模块的输出


# MBConv模块层
class MBConv(layers.Layer):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,  # 输入通道数
                 out_c: int,  # 输出通道数
                 expand_ratio: int,  # 升维倍率因子
                 stride: int,  # 步长
                 se_ratio: float = 0.25,  # SE模块第一层的通道压缩因子
                 drop_rate: float = 0.,  # 主分支随机失活值
                 name: str = None):
        super(MBConv, self).__init__(name=name)

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        # 是否拥有跳链分支
        self.has_shortcut = (stride == 1 and input_c == out_c)
        expanded_c = input_c * expand_ratio  # 根据倍率因子计算升维后通道数
        # 自动生成BN层和卷积层名称
        bid = itertools.count(0)
        get_norm_name = lambda: 'batch_normalization' + ('' if not next(
            bid) else '_' + str(next(bid) // 2))
        cid = itertools.count(0)
        get_conv_name = lambda: 'conv2d' + ('' if not next(cid) else '_' + str(
            next(cid) // 2))

        # EfficientNetV2的MBConv模块层不存在expansion=1的情况，为4或6
        assert expand_ratio != 1
        # MBConv的第一层，升维卷积层
        self.expand_conv = layers.Conv2D(
            filters=expanded_c,  # 卷积核（过滤器）数量
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            name=get_conv_name())
        self.norm0 = layers.BatchNormalization(name=get_norm_name())
        self.act0 = layers.Activation("swish")

        # MBConv的第二层，深度可分离卷积层
        self.depthwise_conv = layers.DepthwiseConv2D(
            kernel_size=kernel_size,  # 卷积核
            strides=stride,  # 步长
            depthwise_initializer=conv_kernel_initializer,
            padding="same",
            use_bias=False,
            name="depthwise_conv2d")
        self.norm1 = layers.BatchNormalization(name=get_norm_name())
        self.act1 = layers.Activation("swish")

        # MBConv的第三层，SE层
        num_reduced_filters = max(1, int(input_c * se_ratio))
        self.se = SE(num_reduced_filters, expanded_c, name="se")

        # MBConv的第四层，降维卷积层，后面只有BN，无激活函数
        self.project_conv = layers.Conv2D(
            filters=out_c,  # 输出通道数
            kernel_size=1,
            strides=1,
            kernel_initializer=conv_kernel_initializer,
            padding="same",
            use_bias=False,
            name=get_conv_name())
        self.norm2 = layers.BatchNormalization(name=get_norm_name())

        # MBConv的第五层，Dropout层
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            # 主分支随机失活，noise_shape指定为 Stochastic Depth 失活模式
            self.drop_path = layers.Dropout(rate=drop_rate,
                                            noise_shape=(None, 1, 1, 1),
                                            name="drop_path")

    # MBConv模块实现逻辑
    def call(self, inputs, training=None, **kwargs):
        x = inputs  # 模块的输入
        x = self.expand_conv(x)  # 升维
        x = self.norm0(x, training=training)
        x = self.act0(x)

        x = self.depthwise_conv(x)  # 深度可分离卷积
        x = self.norm1(x, training=training)
        x = self.act1(x)

        x = self.se(x)  # SE注意力层

        x = self.project_conv(x)  # 降维层
        x = self.norm2(x, training=training)

        if self.has_shortcut:  # Dropout层
            if self.drop_rate > 0:
                x = self.drop_path(x, training=training)
            x = tf.add(x, inputs)

        return x


# Fused-MBConv模块层
class FusedMBConv(layers.Layer):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,  # 升维倍率因子
                 stride: int,
                 se_ratio: float,  # SE节点压缩因子，没有用
                 drop_rate: float = 0.,
                 name: str = None):
        super(FusedMBConv, self).__init__(name=name)
        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        assert se_ratio == 0.  # SE压缩因子应该为0
        # 是否拥有跳连分支
        self.has_shortcut = (stride == 1 and input_c == out_c)
        self.has_expansion = expand_ratio != 1  # 是否包含升维层
        expanded_c = input_c * expand_ratio  #  升维通道数

        bid = itertools.count(0)
        get_norm_name = lambda: 'batch_normalization' + ('' if not next(
            bid) else '_' + str(next(bid) // 2))
        cid = itertools.count(0)
        get_conv_name = lambda: 'conv2d' + ('' if not next(cid) else '_' + str(
            next(cid) // 2))

        if expand_ratio != 1:  # 普通3x3卷积，升维，Fused-MBConv模块第一层
            self.expand_conv = layers.Conv2D(
                filters=expanded_c,
                kernel_size=kernel_size,
                strides=stride,
                kernel_initializer=conv_kernel_initializer,
                padding="same",
                use_bias=False,
                name=get_conv_name())
            self.norm0 = layers.BatchNormalization(name=get_norm_name())
            self.act0 = layers.Activation("swish")

        # 可能是1x1卷积降维，Fused-MBConv模块第二层
        # 或者是3x3卷积，此时为 Fused-MBConv模块第一层
        self.project_conv = layers.Conv2D(
            filters=out_c,
            kernel_size=1 if expand_ratio != 1 else kernel_size,
            strides=1 if expand_ratio != 1 else stride,
            kernel_initializer=conv_kernel_initializer,
            padding="same",
            use_bias=False,
            name=get_conv_name())
        self.norm1 = layers.BatchNormalization(name=get_norm_name())

        if expand_ratio == 1:  # 升维倍率因子为1时需要跟上激活函数
            self.act1 = layers.Activation("swish")
        # Dropout失活层，Fused-MBConv模块第二层或第三层
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            # 主分支随机失活，noise_shape指定为 Stochastic Depth模式
            self.drop_path = layers.Dropout(rate=drop_rate,
                                            noise_shape=(None, 1, 1, 1),  # binary dropout mask
                                            name="drop_path")

    # Fused-MBConv模块层实现逻辑
    def call(self, inputs, training=None, **kwargs):
        x = inputs
        if self.has_expansion:
            x = self.expand_conv(x)  # 1x1升维层
            x = self.norm0(x, training=training)
            x = self.act0(x)

        x = self.project_conv(x) # 3x3卷积或1x1卷积降维
        x = self.norm1(x, training=training)
        if self.has_expansion is False:  #  是否需要激活函数
            x = self.act1(x)

        if self.has_shortcut:  # Dropout层
            if self.drop_rate > 0:
                x = self.drop_path(x, training=training)

            x = tf.add(x, inputs)

        return x


# EfficientNetV2模型的第一个Stage，Conv3x3卷积，模型输入
class Stem(layers.Layer):
    def __init__(self, filters: int, name: str = None):
        super(Stem, self).__init__(name=name)
        self.conv_stem = layers.Conv2D(
            filters=filters,  # 卷积核数量
            kernel_size=3,
            strides=2,
            kernel_initializer=conv_kernel_initializer,
            padding="same",
            use_bias=False,
            name="conv2d")
        self.norm = layers.BatchNormalization(name="batch_normalization")
        self.act = layers.Activation("swish")

    def call(self, inputs, training=None, **kwargs):
        x = self.conv_stem(inputs)
        x = self.norm(x, training=training)
        x = self.act(x)

        return x


# EfficientNetV2模型的最后一个Stage，Conv1×1 & Pooling & FC，模型输出
class Head(layers.Layer):
    def __init__(self,
                 filters: int = 1280,
                 num_classes: int = 1000,
                 drop_rate: float = 0.,  # Pooling与FC之间的Dropout舍弃值
                 name: str = None):
        super(Head, self).__init__(name=name)
        self.conv_head = layers.Conv2D(   # 1x1卷积
            filters=filters,
            kernel_size=1,
            kernel_initializer=conv_kernel_initializer,
            padding="same",
            use_bias=False,
            name="conv2d")
        self.norm = layers.BatchNormalization(name="batch_normalization")
        self.act = layers.Activation("swish")

        self.avg = layers.GlobalAveragePooling2D()  # 全局平均池化
        # 全连接，分类输出层，注意此处没有softmax，预测时再添加
        self.fc = layers.Dense(num_classes,
                               kernel_initializer=dense_kernel_initializer)
        # 此处为普通的Dropout层，随机舍弃节点而不是分支
        if drop_rate > 0:
            self.dropout = layers.Dropout(drop_rate)
    # Head模块层的实现逻辑
    def call(self, inputs, training=None, **kwargs):
        x = self.conv_head(inputs)  # 1x1卷积
        x = self.norm(x)
        x = self.act(x)
        x = self.avg(x)  # 全局平均池化

        if self.dropout:  # 随机失活部分节点
            x = self.dropout(x, training=training)

        x = self.fc(x)  # 按类别全连接输出
        return x


# 现在可以做集成了，定义V2模型整体结构逻辑
class EfficientNetV2(Model):
    def __init__(self,
                 model_cnf: list,  # 模型参数列表，参见后面的定义
                 num_classes: int = 1000,
                 num_features: int = 1280,  # 模型最后提取的特征数量（通道数）
                 dropout_rate: float = 0.2,  # 输出层特征失活概率
                 drop_connect_rate: float = 0.2,  # 分支失活概率
                 name: str = None):
        super(EfficientNetV2, self).__init__(name=name)

        for cnf in model_cnf:
            assert len(cnf) == 8  # 每个Stage配置参数包含8列

        stem_filter_num = model_cnf[0][4] # 第 1 个Stage的过滤器数量
        self.stem = Stem(stem_filter_num) # 第 1 个 Stage 模块，输入模块

        # 统计模型中包含的 Fused—MBConv 和 MBConv 总层数，下面称为 block
        total_blocks = sum([i[0] for i in model_cnf])
        block_id = 0
        self.blocks = []
        # 从前向后，依次构建每个block
        for cnf in model_cnf:
            repeats = cnf[0]  # 当前 block 需要重复的次数
            op = FusedMBConv if cnf[-2] == 0 else MBConv  # block类型
            for i in range(repeats):
                self.blocks.append(op(kernel_size=cnf[1],
                                      input_c=cnf[4] if i == 0 else cnf[5],
                                      out_c=cnf[5],
                                      expand_ratio=cnf[3],
                                      stride=cnf[2] if i == 0 else 1,
                                      se_ratio=cnf[-1],
                                      drop_rate=drop_connect_rate * block_id / total_blocks,
                                      name="blocks_{}".format(block_id)))
                block_id += 1
        # 最后一个Stage，输出层
        self.head = Head(num_features, num_classes, dropout_rate)

    # 模型结构摘要
    def summary(self, input_shape=(224, 224, 3), **kwargs):
        x = Input(shape=input_shape)
        model = Model(inputs=[x], outputs=self.call(x, training=True))
        return model.summary()

    # EfficientNetV2 模型实现逻辑
    def call(self, inputs, training=None, **kwargs):
        x = self.stem(inputs, training) # 输入层

        # 所有的 block 层，Fused—MBConv 和 MBConv
        for _, block in enumerate(self.blocks):
            x = block(x, training=training)

        x = self.head(x, training=training)  # 输出层

        return x

# EfficientNetV2-S 模型函数
def efficientnetv2_s(num_classes: int = 1000):

    # train_size: 300, eval_size: 384
    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[2, 3, 1, 1, 24, 24, 0, 0],
                    [4, 3, 2, 4, 24, 48, 0, 0],
                    [4, 3, 2, 4, 48, 64, 0, 0],
                    [6, 3, 2, 4, 64, 128, 1, 0.25],
                    [9, 3, 1, 6, 128, 160, 1, 0.25],
                    [15, 3, 2, 6, 160, 256, 1, 0.25]]
    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.2,
                           name="efficientnetv2-s")
    return model


# EfficientNetV2-M 模型函数
def efficientnetv2_m(num_classes: int = 1000):

    # train_size: 384, eval_size: 480
    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[3, 3, 1, 1, 24, 24, 0, 0],
                    [5, 3, 2, 4, 24, 48, 0, 0],
                    [5, 3, 2, 4, 48, 80, 0, 0],
                    [7, 3, 2, 4, 80, 160, 1, 0.25],
                    [14, 3, 1, 6, 160, 176, 1, 0.25],
                    [18, 3, 2, 6, 176, 304, 1, 0.25],
                    [5, 3, 1, 6, 304, 512, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.3,
                           name="efficientnetv2-m")
    return model


# EfficientNetV2-L 模型函数
def efficientnetv2_l(num_classes: int = 1000):

    # train_size: 384, eval_size: 480
    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[4, 3, 1, 1, 32, 32, 0, 0],
                    [7, 3, 2, 4, 32, 64, 0, 0],
                    [7, 3, 2, 4, 64, 96, 0, 0],
                    [10, 3, 2, 4, 96, 192, 1, 0.25],
                    [19, 3, 1, 6, 192, 224, 1, 0.25],
                    [25, 3, 2, 6, 224, 384, 1, 0.25],
                    [7, 3, 1, 6, 384, 640, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.4,
                           name="efficientnetv2-l")
    return model

# 观察模型结构
if __name__ == '__main__':
    m = efficientnetv2_s()
    m.summary(input_shape=(300, 300, 3))

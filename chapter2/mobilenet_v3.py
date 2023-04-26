# 功能：MobileNetV3-Large/Small模型实现
# 参考论文原作者发布的源码和GitHub作者WZMIAOMIAO发布的源码改编
# https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
# https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/tensorflow_classification/Test6_mobilenet
# ==============================================================================

from typing import Union
from functools import partial
from tensorflow.keras import layers, Model


# 确保通道数量是8的整数倍并最接近原始值
def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # 调整后的通道数量如果低于原值的 10%，则增加divisor
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def correct_pad(input_size: Union[int, tuple], kernel_size: int):
    """功能：二维卷积运算的 padding 方法
    参数:
      input_size: 输入特征矩阵的高和宽
      kernel_size: 卷积核的高和宽
    返回:一个元组，表示高度上下、宽度左右的填充方案
    """
    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    kernel_size = (kernel_size, kernel_size)
    # 根据高度、宽度的奇偶性，计算调整幅度
    adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    # 根据卷积核尺寸计算调整幅度
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    # 形成填充方案返回((top_pad, bottom_pad), (left_pad, right_pad))
    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


class HardSigmoid(layers.Layer):  # h-sigmoid激活函数
    def __init__(self, **kwargs):
        super(HardSigmoid, self).__init__(**kwargs)
        self.relu6 = layers.ReLU(6.)

    def call(self, inputs, **kwargs):
        x = self.relu6(inputs + 3) * (1. / 6)
        return x


class HardSwish(layers.Layer):  # h-swish激活函数
    def __init__(self, **kwargs):
        super(HardSwish, self).__init__(**kwargs)
        self.hard_sigmoid = HardSigmoid()

    def call(self, inputs, **kwargs):
        x = self.hard_sigmoid(inputs) * inputs
        return x


# SE注意力机制模块
def _se_block(inputs, filters, prefix, se_ratio=1 / 4.):
    # (batch, height, width, channel) -> (batch, 1, 1, channel)
    x = layers.GlobalAveragePooling2D(keepdims=True,
                                      name=prefix + 'squeeze_excite/AvgPool')(inputs)
    # fc1
    x = layers.Conv2D(filters=_make_divisible(filters * se_ratio),
                      kernel_size=1,
                      padding='same',
                      name=prefix + 'squeeze_excite/Conv')(x)
    x = layers.ReLU(name=prefix + 'squeeze_excite/Relu')(x)

    # fc2
    x = layers.Conv2D(filters=filters,
                      kernel_size=1,
                      padding='same',
                      name=prefix + 'squeeze_excite/Conv_1')(x)
    x = HardSigmoid(name=prefix + 'squeeze_excite/HardSigmoid')(x)

    x = layers.Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, x])
    return x


# 反向线性残差模块
def _inverted_res_block(x,                 # 输入的特征矩阵
                        input_c: int,      # 输入的通道数
                        kernel_size: int,  # 卷积核尺寸
                        exp_c: int,        # 扩展后的通道数
                        out_c: int,        # 输出的通道数
                        use_se: bool,      # 是否采用 SE 模块
                        activation: str,   # 激活函数类型：RE 或 HS
                        stride: int,       # 步长
                        block_id: int,     # 残差块编号，共15个模块
                        alpha: float = 1.0 # 宽度调整系数
                        ):
    # BN层函数
    bn = partial(layers.BatchNormalization, epsilon=0.001, momentum=0.99)
    # 调整为最接近原始值的 8 的倍数
    input_c = _make_divisible(input_c * alpha)
    exp_c = _make_divisible(exp_c * alpha)
    out_c = _make_divisible(out_c * alpha)
    # 定义激活函数
    act = layers.ReLU if activation == "RE" else HardSwish

    shortcut = x  # 跳连传递的输入值
    prefix = 'expanded_conv/'
    if block_id:  # 第 1- 14 残差块，不包括索引编号为 0 的残差块
        # 1x1卷积，升维
        prefix = 'expanded_conv_{}/'.format(block_id)
        x = layers.Conv2D(filters=exp_c,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          name=prefix + 'expand')(x)
        x = bn(name=prefix + 'expand/BatchNorm')(x)
        x = act(name=prefix + 'expand/' + act.__name__)(x)

    if stride == 2:  # 步长为2时，对输入的 x 做填充
        input_size = (x.shape[1], x.shape[2])  # height, width
        # ((top_pad, bottom_pad), (left_pad, right_pad))
        x = layers.ZeroPadding2D(padding=correct_pad(input_size, kernel_size),
                                 name=prefix + 'depthwise/pad')(x)
    # 深度可分离卷积
    x = layers.DepthwiseConv2D(kernel_size=kernel_size,
                               strides=stride,
                               padding='same' if stride == 1 else 'valid',
                               use_bias=False,
                               name=prefix + 'depthwise')(x)
    x = bn(name=prefix + 'depthwise/BatchNorm')(x)
    x = act(name=prefix + 'depthwise/' + act.__name__)(x)

    if use_se:  # 采用 SE 模块
        x = _se_block(x, filters=exp_c, prefix=prefix)
    # 1x1卷积，降维
    x = layers.Conv2D(filters=out_c,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      name=prefix + 'project')(x)
    x = bn(name=prefix + 'project/BatchNorm')(x)

    if stride == 1 and input_c == out_c:  # 跳连分支相加
        x = layers.Add(name=prefix + 'Add')([shortcut, x])

    return x


# MobileNetV3-Large模型定义
def mobilenet_v3_large(input_shape=(224, 224, 3),
                       num_classes=1000,
                       alpha=1.0,
                       include_top=True):
    """
    可以从论文官网下载ImageNet预训练权重:
    链接: https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
    """
    bn = partial(layers.BatchNormalization, epsilon=0.001, momentum=0.99)
    img_input = layers.Input(shape=input_shape)
    # 第 1 层
    x = layers.Conv2D(filters=16,
                      kernel_size=3,
                      strides=(2, 2),
                      padding='same',
                      use_bias=False,
                      name="Conv")(img_input)
    x = bn(name="Conv/BatchNorm")(x)
    x = HardSwish(name="Conv/HardSwish")(x)
    # 第 2-16层，即反向线性残差块 0-14
    inverted_cnf = partial(_inverted_res_block, alpha=alpha)
    # input, input_c, k_size, expand_c, output_c, use_se, activation, stride, block_id
    x = inverted_cnf(x, 16, 3, 16, 16, False, "RE", 1, 0)
    x = inverted_cnf(x, 16, 3, 64, 24, False, "RE", 2, 1)
    x = inverted_cnf(x, 24, 3, 72, 24, False, "RE", 1, 2)
    x = inverted_cnf(x, 24, 5, 72, 40, True, "RE", 2, 3)
    x = inverted_cnf(x, 40, 5, 120, 40, True, "RE", 1, 4)
    x = inverted_cnf(x, 40, 5, 120, 40, True, "RE", 1, 5)
    x = inverted_cnf(x, 40, 3, 240, 80, False, "HS", 2, 6)
    x = inverted_cnf(x, 80, 3, 200, 80, False, "HS", 1, 7)
    x = inverted_cnf(x, 80, 3, 184, 80, False, "HS", 1, 8)
    x = inverted_cnf(x, 80, 3, 184, 80, False, "HS", 1, 9)
    x = inverted_cnf(x, 80, 3, 480, 112, True, "HS", 1, 10)
    x = inverted_cnf(x, 112, 3, 672, 112, True, "HS", 1, 11)
    x = inverted_cnf(x, 112, 5, 672, 160, True, "HS", 2, 12)
    x = inverted_cnf(x, 160, 5, 960, 160, True, "HS", 1, 13)
    x = inverted_cnf(x, 160, 5, 960, 160, True, "HS", 1, 14)
    # 第 17 层通道数， 残差块后面的第一个1x1卷积
    last_c = _make_divisible(160 * 6 * alpha)
    x = layers.Conv2D(filters=last_c,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      name="Conv_1")(x)
    x = bn(name="Conv_1/BatchNorm")(x)
    x = HardSwish(name="Conv_1/HardSwish")(x)

    if include_top is True:  # 包含顶层，从池化层开始到最后的分类输出
        # 第 18 层，全局平均池化，注意 keepdims=True ，保持维度
        x = layers.GlobalAveragePooling2D(keepdims=True)(x)
        # fc1 ：第 19 层， 顶层的第 1 个1x1卷积层，相当于全连接层
        last_point_c = _make_divisible(1280 * alpha)
        x = layers.Conv2D(filters=last_point_c,
                          kernel_size=1,
                          padding='same',
                          name="Conv_2")(x)
        x = HardSwish(name="Conv_2/HardSwish")(x)

        # fc2 ： 第20层，顶层的第 1 个1x1卷积层，相当于全连接层
        x = layers.Conv2D(filters=num_classes,
                          kernel_size=1,
                          padding='same',
                          name='Logits/Conv2d_1c_1x1')(x)
        x = layers.Flatten()(x)
        x = layers.Softmax(name="Predictions")(x) # 激活函数为Softmax
    # 组装模型，从输入到输出的逻辑打包
    model = Model(img_input, x, name="MobilenetV3large")

    return model


# MobileNetV3-Small模型定义
def mobilenet_v3_small(input_shape=(224, 224, 3),
                       num_classes=1000,
                       alpha=1.0,
                       include_top=True):
    """
    可以从论文官网下载ImageNet预训练权重:
    链接: https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
    """
    bn = partial(layers.BatchNormalization, epsilon=0.001, momentum=0.99)
    img_input = layers.Input(shape=input_shape)
    # 第 1 层
    x = layers.Conv2D(filters=16,
                      kernel_size=3,
                      strides=(2, 2),
                      padding='same',
                      use_bias=False,
                      name="Conv")(img_input)
    x = bn(name="Conv/BatchNorm")(x)
    x = HardSwish(name="Conv/HardSwish")(x)
    # 反向线性残差模块，从 0 - 10，共 11 个
    inverted_cnf = partial(_inverted_res_block, alpha=alpha)
    # input, input_c, k_size, expand_c, use_se, activation, stride, block_id
    x = inverted_cnf(x, 16, 3, 16, 16, True, "RE", 2, 0)
    x = inverted_cnf(x, 16, 3, 72, 24, False, "RE", 2, 1)
    x = inverted_cnf(x, 24, 3, 88, 24, False, "RE", 1, 2)
    x = inverted_cnf(x, 24, 5, 96, 40, True, "HS", 2, 3)
    x = inverted_cnf(x, 40, 5, 240, 40, True, "HS", 1, 4)
    x = inverted_cnf(x, 40, 5, 240, 40, True, "HS", 1, 5)
    x = inverted_cnf(x, 40, 5, 120, 48, True, "HS", 1, 6)
    x = inverted_cnf(x, 48, 5, 144, 48, True, "HS", 1, 7)
    x = inverted_cnf(x, 48, 5, 288, 96, True, "HS", 2, 8)
    x = inverted_cnf(x, 96, 5, 576, 96, True, "HS", 1, 9)
    x = inverted_cnf(x, 96, 5, 576, 96, True, "HS", 1, 10)
    # 第 13 层 ，残差块后的第一个1x1卷积
    last_c = _make_divisible(96 * 6 * alpha)
    x = layers.Conv2D(filters=last_c,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      name="Conv_1")(x)
    x = bn(name="Conv_1/BatchNorm")(x)
    x = HardSwish(name="Conv_1/HardSwish")(x)

    if include_top is True:  # 包含顶层
        # 第14层，全局平均池化，保持维度
        x = layers.GlobalAveragePooling2D(keepdims=True)(x)

        # fc1：第15层，顶层的第一个1x1卷积，相当于全连接
        last_point_c = _make_divisible(1024 * alpha)
        x = layers.Conv2D(filters=last_point_c,
                          kernel_size=1,
                          padding='same',
                          name="Conv_2")(x)
        x = HardSwish(name="Conv_2/HardSwish")(x)

        # fc2： 第16层，顶层的第二个1x1卷积，相当于全连接
        x = layers.Conv2D(filters=num_classes,
                          kernel_size=1,
                          padding='same',
                          name='Logits/Conv2d_1c_1x1')(x)
        x = layers.Flatten()(x)
        x = layers.Softmax(name="Predictions")(x) # Softmax激活函数
    # 组装模型，从输入到输出逻辑打包
    model = Model(img_input, x, name="MobilenetV3large")

    return model

if __name__ == '__main__':
    model_large = mobilenet_v3_large()
    model_large.summary()
    model_small = mobilenet_v3_small()
    model_small.summary()


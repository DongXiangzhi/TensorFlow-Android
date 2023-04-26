# =====================================================
# faceGenerate.py
# 功能：基于StyleGAN2预训练模型，人脸生成测试
# 设计： 董相志
# 日期： 2022.5.6
# =====================================================
import numpy as np
import matplotlib.pyplot as plt

from utils.utils_stylegan2 import convert_images_to_uint8
from stylegan2_generator import StyleGan2Generator

# 调用生成器随机生成人脸图像并绘图显示
def generate_and_plot_images(gen, seed, w_avg, truncation_psi=1):

    fig, ax = plt.subplots(2, 3, figsize=(15, 15))
    plt.subplots_adjust(wspace=0.1, hspace=0)

    for row in range(2):
        for col in range(3):
            # 初始化随机隐空间向量 z
            rnd = np.random.RandomState(seed)
            z = rnd.randn(1, 512).astype('float32')

            # 运行 隐空间映射网络 mapping network，将 z 映射为 w
            dlatents = gen.mapping_network(z)
            # 根据参数 truncation_psi 调整截断空间
            dlatents = w_avg + (dlatents - w_avg) * truncation_psi
            # 运行合成网络 synthesis network
            out = gen.synthesis_network(dlatents)

            # 将图像数据转换为  uint8 类型，以便于显示
            img = convert_images_to_uint8(out, nchw_to_nhwc=True, uint8_cast=True)

            # 绘制图像
            ax[row,col].axis('off')
            img_plot = ax[row,col].imshow(img.numpy()[0])
            seed += np.random.randint(0,10000)  # 更新seed，保证每幅图像的随机隐向量z不同
    plt.show()


impl = 'ref' # 如果配置了cuda，则用 'cuda'替代
gpu = False  # 如果用GPU，设置为 True

# 加载 ffhq stylegan2 预训练模型
weights_name = 'ffhq' # Nvidia 发布的人脸预训练模型

# 初始化生成器网络
generator = StyleGan2Generator(weights=weights_name, impl=impl, gpu=gpu)

# 加载 权重 w 的平均权重
w_average = np.load('weights/{}_dlatent_avg.npy'.format(weights_name))

seed = np.random.randint(0,100000)
# 不使用截断参数，生成人脸
generate_and_plot_images(generator, seed=seed, w_avg=w_average)

# 设置截断参数为 0.5，生成人脸
generate_and_plot_images(generator, seed=seed, w_avg=w_average, truncation_psi=0.5)

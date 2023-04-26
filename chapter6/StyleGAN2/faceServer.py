# =====================================================
# faceServer.py
# 功能：根据客户机请求，随机生成人脸并发送给客户机，多线程支持并发
# 设计： 董相志
# 日期： 2022.5.6
# =====================================================

import json  #消息头用json格式
import socket
import threading
import numpy as np
from utils.utils_stylegan2 import convert_images_to_uint8
from stylegan2_generator import StyleGan2Generator

MSG_HEADER_LEN = 128  # 用128字节定义消息的长度

# 启动服务器
server_ip = socket.gethostbyname(socket.gethostname()) #获取本机IP
server_port = 50050
server_addr = (server_ip, server_port)
# 创建TCP通信套接字
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(server_addr) #绑定到工作地址
server_socket.listen()  # 开始侦听
print(f'服务器开始在{server_addr}侦听...')
impl = 'ref'  # 如果配置了cuda，则用 'cuda'替代
gpu = False  # 如果用GPU，设置为 True
# 加载 ffhq stylegan2 预训练模型
weights_name = 'ffhq'  # Nvidia 发布的人脸预训练模型
# 初始化生成器网络
generator = StyleGan2Generator(weights=weights_name, impl=impl, gpu=gpu)
# 加载 权重 w 的平均权重
w_average = np.load('weights/{}_dlatent_avg.npy'.format(weights_name))

# 调用生成器随机生成人脸图像并绘图显示
def generate_random_face(gen, seed, w_avg, truncation_psi=1):
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

    return img.numpy()[0]

def handle_client(client_socket, client_addr, generator, w_average):
    """
    功能：与客户机会话线程
    :param client_socket: 会话套接字
    :param client_addr: 客户机地址
    :param model: 预测模型
    """
    print(f'新连接建立，远程客户机地址是：{client_addr}')
    connected = True
    k = 0  # 统计此次会话向客户发送的图像数量
    while connected:
        # 接收消息
        try:
            msg = client_socket.recv(MSG_HEADER_LEN).decode('utf-8')
            # 解析消息头
            msg_header = json.loads(msg)
            msg_type = msg_header['msg_type']
        except:
            print(f'客户机 {client_addr} 连接异常！')
            break
        # 生成图像
        seed = np.random.randint(0, 10000000)
        # 设置截断参数为 0.5，生成人脸
        face = generate_random_face(generator,
                             seed=seed,
                             w_avg=w_average,
                             truncation_psi=0.5)

        face = face.flatten()  # 变一维数组
        if msg_type == 'GET_FACE': # 收到人脸请求消息
            # 定义消息头
            size = len(face)

            header = {"msg_type": "FACE_IMAGE", "msg_len": size}
            header_byte = bytes(json.dumps(header), encoding='utf-8')
            header_byte += b' ' * (MSG_HEADER_LEN - len(header_byte))  # 消息头补空格
            client_socket.sendall(header_byte)  # 发送消息头
            client_socket.sendall(face)  # 发送消息内容
            k += 1
            print(f"向客户机：{client_addr} 随机发送了第 {k} 幅人脸图像！")
        elif msg_type == 'SHUT_DOWN': # 收到客户机下线消息
            break
    print(f"客户机：{client_addr} 关闭了连接！")
    client_socket.close()  # 关闭会话连接

while True:
    new_socket, new_addr = server_socket.accept() #处理连接
    # 建立与客户机会话的线程，一客户一线程
    client_thread = threading.Thread(target=handle_client,
                                     args=(new_socket, new_addr,
                                           generator, w_average))
    client_thread.start()


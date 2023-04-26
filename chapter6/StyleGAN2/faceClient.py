# =====================================================
# faceClient.py
# 功能：桌面版客户机
# 设计： 董相志
# 日期： 2022.5.6
# =====================================================
import json  #消息头用json格式
import socket
import numpy as np
import matplotlib.pyplot as plt

MSG_HEADER_LEN = 128 #用128字节定义消息头的长度

# 服务器地址
server_ip = socket.gethostbyname(socket.gethostname()) #获取本机IP
server_port = 50050
server_addr = (server_ip, server_port)
# 创建TCP通信套接字
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(server_addr)  # 连接服务器

for i in range(20):   # 随机测试20幅图片
    header = {'msg_type':'GET_FACE','msg_len':1}  # 请求图片消息
    header_byte = bytes(json.dumps(header), encoding='utf-8')
    header_byte += b' ' * (MSG_HEADER_LEN - len(header_byte))  # 消息头补空格
    client_socket.sendall(header_byte)  # 发送消息头

    # 接收来自服务器的消息头
    msg_header = client_socket.recv(MSG_HEADER_LEN).decode('utf-8')
    # 解析头部
    header = json.loads(msg_header) #字符串转化为字典
    msg_len = header['msg_len']  # 消息长度

    data = bytearray()
    while len(data) < msg_len:  # 接收图像数据
        bytes_read = client_socket.recv(msg_len - len(data))
        if not bytes_read:
            break
        data.extend(bytes_read)

    # 图像转换
    image = np.array(data).reshape(1024,1024,3)
    plt.imshow(image)
    plt.show()

# 发送下线消息
header = {'msg_type':'SHUT_DOWN','msg_len':1}  # 请求图片消息
header_byte = bytes(json.dumps(header), encoding='utf-8')
header_byte += b' ' * (MSG_HEADER_LEN - len(header_byte))  # 消息头补空格
client_socket.sendall(header_byte)  # 发送下线消息头
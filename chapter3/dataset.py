# ===========================================
# dataset.py
# 功能：对数据集做预处理，划分训练集、验证集和测试集，生成标签文件datasets.csv
# 设计： 董相志
# 日期： 2022.2.27
# ===========================================
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from PIL import Image
all_foods = []  # 存放所有样本标签
# 读取所有类别名称
category = pd.read_table('./dataset100/category.txt')
# 列表中列的顺序
column_order = ['type', 'img', 'label', 'x1', 'y1', 'x2', 'y2']
# 遍历目录 1~100，读取所有图片的标签信息，汇集到all_foods列表
for i in range(1,101,1):
    # 读取当前目录i的标签信息
    foods = pd.read_table(f'./dataset100/{i}/bb_info.txt',
                          header=0,
                          sep='\s+')
    # 将图像id映射为对应的文件路径
    foods['img'] = foods['img'].apply(lambda x: f'./dataset100/{i}/' + str(x) +'.jpg')
    # 新增一列 label，标注图片类别名称
    foods['label'] = foods.apply(lambda x: category['name'][i-1], axis=1)
    foods['type'] = foods.apply(lambda x: '', axis=1)
    foods = foods[column_order]
    # 保存当前类别的标签文件
    foods.to_csv(f'./dataset100/{i}/label.csv',
                 index=None,
                 header=['type', 'img', 'label', 'x1', 'y1', 'x2', 'y2'])
    # 汇聚到列表 all_foods
    all_foods.extend(np.array(foods).tolist())

# 保存列表到文件中
df_foods = pd.DataFrame(all_foods)
df_foods.to_csv('./dataset100/all_foods.csv',
                index=None,
                header=['type', 'img', 'label', 'x1', 'y1', 'x2', 'y2'])
# 随机洗牌，打乱数据集排列顺序，划分为 TRAIN、VALIDATE、TEST三部分
datasets = pd.read_csv('./dataset100/all_foods.csv')  # 读数据
datasets = shuffle(datasets,random_state=2022)  # 洗牌
datasets = pd.DataFrame(datasets).reset_index(drop=True)
rows = datasets.shape[0]  # 总行数
test_n = rows // 40  # 测试集样本数
validate_n = rows // 5  # 验证集样本数
train_n = rows - test_n - validate_n  # 训练集样本数
print(f'测试集样本数：{test_n},验证集样本数：{validate_n},训练集样本数：{train_n}')
# 按照一定比例对数据集划分
for row in range(test_n):  # 标注测试集
    datasets.iloc[row, 0] = 'TEST'
for row in range(validate_n):  # 标注验证集
    datasets.iloc[row + test_n, 0] = 'VALIDATE'
for row in range(train_n):  # 标注训练集
    datasets.iloc[row + test_n + validate_n, 0] = 'TRAIN'

# 将bounding box的坐标改为浮点类型,取值范围为[0,1]
print('开始对BBox坐标做归一化调整，请耐心等待...')
for row in range(rows):
    img = Image.open(datasets.iloc[row, 1])  # 读取图像
    (width, height) = img.size  # 图像宽度与高度
    width = float(width)
    height = float(height)
    datasets.iloc[row, 3] = round(datasets.iloc[row, 3] / width, 3)
    datasets.iloc[row, 4] = round(datasets.iloc[row, 4] / height, 3)
    datasets.iloc[row, 5] = round(datasets.iloc[row, 5] / width, 3 )
    datasets.iloc[row, 6] = round(datasets.iloc[row, 6] / height, 3)
datasets.insert(datasets.shape[1], 'Null1', '')  # 插入空列
datasets.insert(datasets.shape[1], 'Null2', '')  # 插入空列
# 调整列的顺序，为以后数据集划分做准备
order = ['type', 'img', 'label', 'x1', 'y1', 'Null1', 'Null2', 'x2', 'y2']
datasets = datasets[order]
print(datasets.head())
datasets.to_csv('./dataset100/datasets.csv', index=None, header=None)
print('数据集构建完毕！')
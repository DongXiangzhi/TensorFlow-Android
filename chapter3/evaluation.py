# ===========================================
# evaluation.py
# 功能：根据AP值分类别绘制模型条形图
# 设计： 董相志
# 日期： 2022.3.6
# ===========================================
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 读取电脑版TFLite的模型评估数据
dict1 = [json.loads(line) for line in open(r'dict1.txt','r')]
for key in dict1[0]:
    dict1[0][key] = float(dict1[0][key])
df1 = pd.DataFrame(dict1)
print(df1.head())

# 读取移动版TFLite模型评估数据
dict2 = [json.loads(line) for line in open(r'dict2.txt','r')]
for key in dict2[0]:
    dict2[0][key] = float(dict2[0][key])
df2 = pd.DataFrame(dict2)
print(df2.head())

# 取前12项指标
columns = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',
           'ARmax1', 'ARmax10', 'ARmax100', 'ARs','ARm','ARl']
df1_12 = df1.iloc[0, 0:12]
df2_12 = df2.iloc[0, 0:12]
sns.barplot(x=np.array(df1_12).tolist(), y=columns)  # 电脑版TFLite
plt.show()
sns.barplot(x=np.array(df2_12).tolist(), y=columns)  # 移动版TFLite
plt.show()

# 100个类别mAP指标的条形图
df1.drop(columns=columns, inplace=True, axis=1)
df1 = df1.stack()  # 行列互换
df1 = df1.unstack(0)
df1.sort_values(by=0, axis=0, ascending=False, inplace=True)
df2.drop(columns=columns, inplace=True, axis=1)
df2 = df2.stack()  # 行列互换
df2 = df2.unstack(0)
df2.sort_values(by=0, axis=0, ascending=False, inplace=True)
# 根据需要，只显示mAP值最高的前20个类别
sns.barplot(x=df1[0][0:20], y=df1.index[0:20])  # 电脑版TFLite
plt.show()
sns.barplot(x=df2[0][0:20], y=df2.index[0:20])  # 移动版TFLite
plt.show()
# 只显示mAP值最低的20个类别
sns.barplot(x=df1[0][-20:], y=df1.index[-20:])  # 电脑版TFLite
plt.show()
sns.barplot(x=df2[0][-20:], y=df2.index[-20:])  # 移动版TFLite
plt.show()






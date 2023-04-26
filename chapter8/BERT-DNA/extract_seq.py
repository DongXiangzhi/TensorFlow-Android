# ==================================
# 程序：extract_seq.py
# 功能：DNA序列生成
# 日期：2022.2.20
# 设计：董相志
# 源文件：data目录下
# ==================================

import os
import re

full_path = os.path.realpath(__file__)
os.chdir(os.path.dirname(full_path))


# 序列生成函数
def extract_seq(old_file_path, new_file_path='seqs', seq_length=512):
    if not os.path.exists('seqs'):
        os.makedirs('seqs')
    nseq = 0  # 序列计数
    nsmp = 0  # 样本计数
    all_sequences = []  # 所有的样本序列
    data = re.split(
        r'(^>.*)', ''.join(open(old_file_path).readlines()), flags=re.M)
    for i in range(2, len(data), 2):
        nseq = nseq + 1
        # 生成序列，核苷酸之间添加空格
        fasta = list(data[i].replace('\n', '').replace('\x1a', ''))
        seq = [' '.join(fasta[j:j + seq_length])
               for j in range(0, len(fasta) + 1, seq_length)]
        nsmp = nsmp + len(seq)
        all_sequences.append('\n'.join(seq))  # 样本序列加到列表
    # 序列保存为独立文件
    with open(f"./seqs/{new_file_path}.seq", "w") as ffas:
        ffas.write('\n'.join(all_sequences))
    print(f"文件 {old_file_path} 包含的序列数量: {nseq}")
    print(f"文件 {old_file_path} 包含的样本数量: {nsmp}")

# 提取DNA序列，并且在碱基字母间添加空格，保存到新文件中
extract_seq('./data/enhancer.cv.txt', 'cv_pos') # 训练集增强子序列正样本
extract_seq('./data/non.cv.txt', 'cv_neg') # 训练集增强子序列负样本
extract_seq('./data/enhancer.ind.txt', 'ind_pos') # 测试集增强子序列正样本
extract_seq('./data/non.ind.txt', 'ind_neg') # 测试集增强子序列负样本


# ===========================================
# bert_feature.py
# 功能：用 BERT 模型提取DNA序列特征，特征和标签存入hdf5文件
# 设计： 董相志
# 日期： 2022.2.20
# ===========================================

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import h5py

# BERT预训练模型下载地址
# BERT_MODEL = "https://tfhub.dev/google/experts/bert/wiki_books/2"
BERT_MODEL = "./BERT-hub/experts_bert_wiki_books_2"
# 序列输入BERT模型之前的预处理，预处理模型下载地址
# PREPROCESS_MODEL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
PREPROCESS_MODEL = "./BERT-hub/bert_en_uncased_preprocess_3"

# 读取样本数据集文件，返回样本列表和长度
def read_samples(filepath):
    # 打开样本训练集文件
    with open(filepath,'r') as f:
        dna_sentences = f.readlines()
    length = len(dna_sentences)  # 样本数量
    # 去掉其中的换行符
    for i in range(length):
        dna_sentences[i] = dna_sentences[i].replace('\n','')
    return dna_sentences,length  # 返回序列集列表和长度


# 读取训练集正样本列表
train_pos_dna_sentences,train_pos_len = read_samples('./seqs/cv_pos.seq')
print(train_pos_dna_sentences[0:2])
# 读取训练集负样本列表
train_neg_dna_sentences,train_neg_len = read_samples('./seqs/cv_neg.seq')
# 读取测试集正样本列表
test_pos_dna_sentences,test_pos_len = read_samples('./seqs/ind_pos.seq')
# 读取测试集负样本列表
test_neg_dna_sentences,test_neg_len = read_samples('./seqs/ind_neg.seq')

train_samples = train_pos_len + train_neg_len  # 训练集样本总数
test_samples = test_pos_len + test_neg_len  # 测试集样本总数

X_train = np.zeros((train_samples,200,768))  # 初始化训练集特征矩阵
y_train = np.zeros((train_samples,1))  # 初始化训练集标签矩阵
X_test = np.zeros((test_samples,200,768))  # 初始化测试集特征矩阵
y_test = np.zeros((test_samples,1))  # 初始化测试集标签矩阵

# 加载DNA序列的预处理模型
preprocessor = hub.load(PREPROCESS_MODEL)
# 定义输入层
text_inputs = [tf.keras.layers.Input(shape=(),dtype=tf.string)]
# 得到序列分词列表
tokenize = hub.KerasLayer(preprocessor.tokenize)
tokenized_inputs = [tokenize(segment) for segment in text_inputs]

seq_length=202  # 设定输入的序列最大长度，200+CLS+SEP
# 序列分词列表转化为BERT模型的输入向量
bert_pack_inputs = hub.KerasLayer(preprocessor.bert_pack_inputs,
                                  arguments=dict(seq_length=seq_length))
encoder_inputs = bert_pack_inputs(tokenized_inputs)
# 以下代码块定义BERT模型
encoder = hub.KerasLayer(BERT_MODEL,trainable=False)  # 用BERT模型编码
outputs = encoder(encoder_inputs)
# [batch_size, 768]，模型的最终输出
pooled_output = outputs["pooled_output"]
# [batch_size, seq_length,768]，序列编码输出
sequence_output = outputs["sequence_output"]
# 定义输出特征的BERT模型，注意这里需要用sequence_output
bert_model = tf.keras.Model(text_inputs,sequence_output)
print('数据集特征提取的时间与数据集规模以及计算力相关，可能需要数分钟，请耐心等待...')
# 调用BERT模型，完成序列编码，存储到hdf5格式的文件作为数据集
# 对训练集正样本编码
for i in range(train_pos_len):
    dna_sentence = tf.constant(train_pos_dna_sentences[i])
    dna_sentence = np.expand_dims(dna_sentence,axis=0)
    dna_feature = bert_model(dna_sentence)  # 特征向量 [1, 202,768]
    dna_feature = dna_feature[:,1:201,:] # 去掉CLS和SEP，新维度[1,200,768]
    # 写入训练集矩阵
    X_train[i] = dna_feature  # DNA序列的特征
    y_train[i] = 1  # 表示正样本标签，对应增强子
# 对训练集负样本编码
for i in range(train_neg_len):
    dna_sentence = tf.constant(train_neg_dna_sentences[i])
    dna_sentence = np.expand_dims(dna_sentence,axis=0)
    dna_feature = bert_model(dna_sentence)  # 特征向量 [1, 202,768]
    dna_feature = dna_feature[:, 1:201, :]  # 去掉CLS和SEP，维度[1,200,768]
    # 写入训练集矩阵
    X_train[train_pos_len+i] = dna_feature  # DNA序列的特征
    y_train[train_pos_len+i] = 0  # 表示负样本标签，对应非增强子

# 将训练集数据写到hdf5文件中
file_train = h5py.File('./data/dna_train.hdf5','w')
dataset = file_train.create_dataset("X_train", data = X_train)
dataset = file_train.create_dataset("y_train", data = y_train)
file_train.close()

# 对测试练集正样本编码
for i in range(test_pos_len):
    dna_sentence = tf.constant(test_pos_dna_sentences[i])
    dna_sentence = np.expand_dims(dna_sentence,axis=0)
    dna_feature = bert_model(dna_sentence)  # 特征向量 [1, 202,768]
    dna_feature = dna_feature[:, 1:201, :] # 去掉CLS和SEP，维度[1,200,768]
    # 写入测试集矩阵
    X_test[i] = dna_feature  # DNA序列的特征
    y_test[i] = 1  # 表示正样本标签，对应增强子
# 对测试集负样本编码
for i in range(test_neg_len):
    dna_sentence = tf.constant(test_neg_dna_sentences[i])
    dna_sentence = np.expand_dims(dna_sentence,axis=0)
    dna_feature = bert_model(dna_sentence)  # 特征向量 [1, 202,768]
    dna_feature = dna_feature[:,1:201,:] # 去掉CLS和SEP，新维度[1,200,768]
    # 写入测试集矩阵
    X_train[test_pos_len+i] = dna_feature  # DNA序列的特征
    y_train[test_pos_len+i] = 0  # 表示负样本标签，对应非增强子

# 将测试集数据写到hdf5文件中
file_test = h5py.File('./data/dna_test.hdf5','w')
dataset = file_test.create_dataset("X_test", data = X_test)
dataset = file_test.create_dataset("y_test", data = y_test)
file_test.close()




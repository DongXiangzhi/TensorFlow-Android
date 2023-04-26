'''
模块名：dataset.py
功 能：构建训练集与验证集
设 计：董相志
日 期：2021.7
'''

import codecs
import json
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 析取数据集（训练集、验证集、测试集），将所有的“问”与“答”分开
def extract_conversations(hparams, data_list, dialog_list):
  inputs, outputs = [], []  # 问答列表
  for dialog in dialog_list:
    if dialog['dialog_id'] in data_list:
      if len(dialog['content']) % 2 == 0:
        i = 0
        for line in dialog['content']:
          if (i % 2 == 0):
            inputs.append(line)  # “问”列表
          else:
            outputs.append(line)  # “答”列表
          i += 1
          # 限定样本总数
          # if len(inputs) >= hparams.total_samples:
          #     return inputs, outputs
  return inputs, outputs

# 分词，过滤掉超过长度的句子，短句补齐
def tokenize_and_filter(hparams, inputs, outputs, tokenizer):
    tokenized_inputs, tokenized_outputs = [], []
    for (sentence1, sentence2) in zip(inputs, outputs):
        sentence1 = tokenizer.tokenize(sentence1)  # 分词
        sentence1 = tokenizer.convert_tokens_to_ids(sentence1)  # ids
        sentence2 = tokenizer.tokenize(sentence2)
        sentence2 = tokenizer.convert_tokens_to_ids(sentence2)
        sentence1 = hparams.start_token + sentence1 + hparams.end_token
        sentence2 = hparams.start_token + sentence2 + hparams.end_token
        if len(sentence1) <= hparams.max_length and \
                len(sentence2) <= hparams.max_length:
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)
    # 补齐
    tokenized_inputs = pad_sequences(tokenized_inputs, \
                            maxlen=hparams.max_length, padding='post')
    tokenized_outputs = pad_sequences(tokenized_outputs, \
                            maxlen=hparams.max_length, padding='post')

    return tokenized_inputs, tokenized_outputs

# 读文件
def get_data(datafile):
    with open(f'{datafile}', 'r') as f:
        data_list = f.readlines()
        for i in range(len(data_list)):
            data_list[i] = re.sub(r'\n', '', data_list[i])
        return data_list

# 返回训练集和和验证集
def get_dataset(hparams, tokenizer, dialog_file, train_file, valid_file):
    dialog_list = json.loads( \
        codecs.open(f"{dialog_file}", "r", "utf-8").read())
    print(dialog_list[0])
    train_list = get_data(f'{train_file}')
    train_questions, train_answers = extract_conversations(hparams, \
                                            train_list, dialog_list)
    train_questions, train_answers = tokenize_and_filter(hparams, \
        list(train_questions), list(train_answers), tokenizer)
    # 构建训练集
    train_dataset = tf.data.Dataset.from_tensor_slices((
      {
          'inputs': train_questions,
          # 解码器使用正确的标签做为输入
          'dec_inputs': train_answers[:, :-1]  # 去掉最后一个元素或 END_TOKEN
      },
      {
          'outputs': train_answers[:, 1:]  # 去掉 START_TOKEN
      },
    ))
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(len(train_questions))
    train_dataset = train_dataset.batch(hparams.batchSize)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    # 构建验证集
    valid_list = get_data(f'{valid_file}')
    valid_questions, valid_answers = extract_conversations( \
                     hparams,valid_list, dialog_list)
    valid_questions, valid_answers = tokenize_and_filter(hparams,  \
        list(valid_questions), list(valid_answers), tokenizer)
    valid_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': valid_questions,
            # 解码器使用正确的标签做为输入
            'dec_inputs': valid_answers[:, :-1]  # 去掉最后一个元素或 END_TOKEN
        },
        {
            'outputs': valid_answers[:, 1:]  # 去掉START_TOKEN
        },
    ))
    valid_dataset = valid_dataset.cache()
    valid_dataset = valid_dataset.shuffle(len(valid_questions))
    valid_dataset = valid_dataset.batch(hparams.batchSize)
    valid_dataset = valid_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, valid_dataset

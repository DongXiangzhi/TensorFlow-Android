'''
模块名：main.py
功 能：人机聊天主程序，根据dataset、model两个模块，完成模型训练
设 计：董相志
日 期：2021.7
'''
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow import multiply, minimum
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.metrics import sparse_categorical_accuracy
from tensorflow.python.ops.math_ops import rsqrt
from tensorflow.keras.optimizers import Adam
from bert.tokenization.bert_tokenization import FullTokenizer
import numpy as np
# 用BLEU方法评估模型
from nltk.translate.bleu_score import sentence_bleu

from transformer.model import *
from transformer.dataset import *

if __name__ == "__main__" :
    dialog_list = json.loads( \
        codecs.open("transformer/dataset/dialog_release.json", \
        "r", "utf-8").read())
    print(dialog_list[0]) # 第一条数据

    # 以下参数可根据需要调整，为了演示之目的，可以将相关参数调低一些。
    # 最大句子长度
    MAX_LENGTH = 40

    # 最大样本数量
    MAX_SAMPLES = 120000  # 可根据需要调节

    BATCH_SIZE = 64  # 批处理大小

    # Transformer参数定义
    NUM_LAYERS = 2  # 编码器解码器block重复数，论文中是6
    D_MODEL = 128  # 编码器解码器宽度，论文中是512
    NUM_HEADS = 4  # 注意力头数，论文中是8
    UNITS = 512  # 全连接网络宽度，论文中输入输出为512
    DROPOUT = 0.1  # 与论文一致
    VOCAB_SIZE = 21128  # BERT词典长度

    START_TOKEN = [VOCAB_SIZE]  # 序列起始标志
    END_TOKEN = [VOCAB_SIZE + 1]  # 序列结束标志
    VOCAB_SIZE = VOCAB_SIZE + 2  # 加上开始与结束标志后的词典长度
    EPOCHS = 50  # 训练代数

    bert_vocab_file = 'transformer/dataset/vocab.txt'
    tokenizer = FullTokenizer(bert_vocab_file)
    # 聊天模型参数配置与结构定义
    model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT)
    model.summary()

    # 学习率动态调度
    class CustomSchedule(LearningRateSchedule):

        def __init__(self, d_model, warmup_steps=4000):
            super(CustomSchedule, self).__init__()

            self.d_model = tf.constant(d_model, dtype=tf.float32)
            self.warmup_steps = warmup_steps

        def get_config(self):
            return {"d_model": self.d_model, "warmup_steps": self.warmup_steps}

        def __call__(self, step):
            arg1 = rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)

            return multiply(rsqrt(self.d_model), minimum(arg1, arg2))

    # 测试
    sample_learning_rate = CustomSchedule(d_model=256)

    plt.plot(sample_learning_rate(tf.range(200000, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.show()

    learning_rate = CustomSchedule(D_MODEL)  # 学习率

    # 定义损失函数
    def loss_function(y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

        loss = SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')(y_true, y_pred)

        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)

        return tf.reduce_mean(loss)


    # 自定义准确率函数
    def accuracy(y_true, y_pred):
        # 调整标签的维度为：(batch_size, MAX_LENGTH - 1)
        y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
        return sparse_categorical_accuracy(y_true, y_pred)

    # 优化算法
    optimizer = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

    # 定义回调函数：保存最优模型
    checkpoint = ModelCheckpoint("robot_weights.h5",
                                 monitor="val_loss",
                                 mode="min",
                                 save_best_only=True,
                                 save_weights_only=True,
                                 verbose=1)
    # 定义回调函数：提前终止训练
    earlystop = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=10,
                              verbose=1,
                              restore_best_weights=True)

    # 将回调函数组织为回调列表
    callbacks = [earlystop, checkpoint]
    dialog_file = 'transformer/dataset/dialog_release.json'
    train_file = 'transformer/dataset/train.txt'
    valid_file = 'transformer/dataset/dev.txt'
    class Hparams() :
        def __init__(self,start_token,end_token,batchSize,total_samples,max_length):
            self.start_token = start_token
            self.end_token = end_token
            self.batchSize = batchSize
            self.total_samples = total_samples
            self.max_length = max_length
    hparams = Hparams
    hparams.start_token = START_TOKEN
    hparams.end_token =END_TOKEN
    hparams.total_samples = MAX_SAMPLES
    hparams.batchSize = BATCH_SIZE
    hparams.max_length = MAX_LENGTH
    # 加载并划分数据集
    train_dataset, valid_dataset = get_dataset(hparams, tokenizer, dialog_file, train_file, valid_file)
    # # 模型训练
    # history = model.fit(train_dataset, epochs=EPOCHS, \
    #                     validation_data=valid_dataset, \
    #                     callbacks=callbacks)
    # # 损失函数曲线
    # plt.figure(figsize=(12, 6))
    # x = range(1, len(history.history['loss']) + 1)
    # plt.plot(x, history.history['loss'])
    # plt.plot(x, history.history['val_loss'])
    # plt.xticks(x)
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['train', 'test'])
    # plt.title('Loss over training epochs')
    # plt.savefig('loss.png')
    # plt.show()
    # # 准确率曲线
    # plt.figure(figsize=(12, 6))
    # plt.plot(x, history.history['accuracy'])
    # plt.plot(x, history.history['val_accuracy'])
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.xticks(x)
    # plt.legend(['train', 'test'])
    # plt.title('Accuracy over training epochs')
    # plt.savefig('acc.png')
    # plt.show()

    # 加载训练好的模型
    model.load_weights('models/robot_weights_l2.h5')

    # 用模型做聊天推理，A、B两人聊天，输入 A 的句子，得到 B 的回应
    def evaluate(sentence):
        sentence = tokenizer.tokenize(sentence)
        sentence = START_TOKEN + tokenizer.convert_tokens_to_ids(sentence) + END_TOKEN
        sentence = tf.expand_dims(sentence, axis=0)

        output = tf.expand_dims(START_TOKEN, 0)

        for i in range(MAX_LENGTH):
            predictions = model(inputs=[sentence, output], training=False)

            # 选择最后一个输出
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # 如果是END_TOKEN则结束预测
            if tf.equal(predicted_id, END_TOKEN[0]):
                break

            # 把已经得到的预测值串联起来，做为解码器的新输入
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0)


    # 模拟聊天间的问答，输入问话，输出回答
    def predict(question):
        prediction = evaluate(question)  # 调用模型推理
        predicted_answer = tokenizer.convert_ids_to_tokens(
            np.array([i for i in prediction if i < VOCAB_SIZE - 2]))

        print(f'问话者: {question}')
        print(f'答话者: {"".join(predicted_answer)}')

        return predicted_answer

    # 几组随机测试
    output1 = predict('嗨，你好呀。')  # 训练集中的样本
    print("")
    output2 = predict('昨晚的比赛你看了吗？')  # 随机问话
    print("")
    output3 = predict('你最喜欢的人是谁？')  # 随机问话
    print("")
    output4 = predict('真热，下点雨儿就好了')  # 随机问话
    print("")
    output5 = predict('这个老师讲课怎么样？')  # 随机问话
    print("")
    output6 = predict('今天收获大吗？')  # 随机问话
    print("")
    # 多轮对话测试，自问自答
    sentence = '你最近有听说过《中国女排》这部电影嘛？'
    for _ in range(5):
        sentence = "".join(predict(sentence))
        print("")


    reference = '是呀，我觉得这部《中国女排》应该能拿下很高的收视率。'
    pred_sentence = predict(reference)
    reference = tokenizer.tokenize(reference)

    # 1-gram BLEU计算
    BLEU_1 = sentence_bleu([reference], pred_sentence, weights=(1, 0, 0, 0))
    print(f"\n BLEU-1 评分: {BLEU_1}")

    # 2-gram BLEU计算
    BLEU_2 = sentence_bleu([reference], pred_sentence, weights=(0.5, 0.5, 0, 0))
    print(f"\n BLEU-2 评分: {BLEU_2}")

    # 3-gram BLEU计算
    BLEU_3 = sentence_bleu([reference], pred_sentence, weights=(0.33, 0.33, 0.33, 0))
    print(f"\n BLEU-3 评分:: {BLEU_3}")

    # 4-gram BLEU计算
    BLEU_4 = sentence_bleu([reference], pred_sentence, weights=(0.25, 0.25, 0.25, 0.25))
    print(f"\n BLEU-4 评分:: {BLEU_4}")

    # 5-gram BLEU计算
    BLEU_5 = sentence_bleu([reference], pred_sentence, weights=(0.2, 0.2, 0.2, 0.2, 0.2))
    print(f"\n BLEU-5 评分:: {BLEU_5}")

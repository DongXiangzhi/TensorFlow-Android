'''
模块名：model.py
功 能：根据论文定义Transformer，结构可调整
设 计：董相志
日 期：2021.7
'''
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Lambda, Embedding, Dropout, \
    add, LayerNormalization
from tensorflow.keras.utils import plot_model

# 定义掩码矩阵
def create_padding_mask(x):
    # 找出序列中的 padding，设置掩码值为 1.
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, sequence length)
    return mask[:, tf.newaxis, tf.newaxis, :]
# 测试语句
# print(create_padding_mask(tf.constant([[1, 2, 0, 3, 0], [0, 0, 0, 4, 5]])))
# 解码器的前向掩码
def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)
# 测试
# print(create_look_ahead_mask(tf.constant([[1, 2, 0, 4, 5]])))

# 位置编码类
class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'position': self.position,
            'd_model': self.d_model,

        })
        return config

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles( \
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis], \
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :], \
            d_model=d_model)
        # 奇数位置用sin函数
        sines = tf.math.sin(angle_rads[:, 0::2])
        # 偶数位置用cos函数
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

# 测试
# sample_pos_encoding = PositionalEncoding(50, 512)
#
# plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
# plt.xlabel('Depth')
# plt.xlim((0, 512))
# plt.ylabel('Position')
# plt.colorbar()
# plt.show()


# 计算注意力
def scaled_dot_product_attention(query, key, value, mask):
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # 计算qk
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # 添加掩码以将填充标记归零
    if mask is not None:
        logits += (mask * -1e9)

    # 在最后一个轴上实施softmax
    attention_weights = tf.nn.softmax(logits, axis=-1)

    output = tf.matmul(attention_weights, value)

    return output

# 定义多头注意力类，继承了Layer类
class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = Dense(units=d_model)
        self.key_dense = Dense(units=d_model)
        self.value_dense = Dense(units=d_model)

        self.dense = Dense(units=d_model)

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            'num_heads':self.num_heads,
            'd_model':self.d_model,
        })
        return config

    def split_heads(self, inputs, batch_size):
        inputs = Lambda(lambda inputs:tf.reshape(inputs, \
                shape=(batch_size, -1, self.num_heads, self.depth)))(inputs)
        return Lambda(lambda inputs: tf.transpose(inputs, \
                                                  perm=[0, 2, 1, 3]))(inputs)

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # 线性层变换
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # 分头
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # 定义缩放的点积注意力
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = Lambda(lambda scaled_attention: tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]))(scaled_attention)

        # 堆叠注意力头
        concat_attention = Lambda(lambda scaled_attention: tf.reshape( \
            scaled_attention,(batch_size, -1, self.d_model)))(scaled_attention)

        # 多头注意力最后一层
        outputs = self.dense(concat_attention)

        return outputs


# 编码器中的一层
def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention = MultiHeadAttention( \
      d_model, num_heads, name="attention")({ \
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': padding_mask
      })
    attention = Dropout(rate=dropout)(attention)
    add_attention = add([inputs,attention])
    attention = LayerNormalization(epsilon=1e-6)(add_attention)

    outputs = Dense(units=units, activation='relu')(attention)
    outputs = Dense(units=d_model)(outputs)
    outputs = Dropout(rate=dropout)(outputs)
    add_attention = add([attention,outputs])
    outputs = LayerNormalization(epsilon=1e-6)(add_attention)

    return Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

# 测试
# sample_encoder_layer = encoder_layer(
#     units=512,
#     d_model=128,
#     num_heads=4,
#     dropout=0.3,
#     name="sample_encoder_layer")
#
# plot_model(sample_encoder_layer, to_file='encoder_layer.png', show_shapes=True)



# 定义编码器
def encoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="encoder"):
    inputs = Input(shape=(None,), name="inputs")
    padding_mask = Input(shape=(1, 1, None), name="padding_mask")

    embeddings = Embedding(vocab_size, d_model)(inputs)
    embeddings *= Lambda(lambda d_model: tf.math.sqrt(tf.cast(d_model, tf.float32)))(d_model)
    embeddings = PositionalEncoding(vocab_size,d_model)(embeddings)

    outputs = Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="encoder_layer_{}".format(i),
        )([outputs, padding_mask])

    return Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)
# 编码器测试
# sample_encoder = encoder(
#     vocab_size=21128,
#     num_layers=2,
#     units=512,
#     d_model=128,
#     num_heads=4,
#     dropout=0.3,
#     name="sample_encoder")
#
# plot_model(sample_encoder, to_file='encoder.png', show_shapes=True)


# 定义解码器中的一层
def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = Input(shape=(None, d_model), name="inputs")
    enc_outputs = Input(shape=(None, d_model), name="encoder_outputs")
    look_ahead_mask = Input(shape=(1, None, None), name="look_ahead_mask")
    padding_mask = Input(shape=(1, 1, None), name='padding_mask')

    attention1 = MultiHeadAttention(
      d_model, num_heads, name="attention_1")(inputs={
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': look_ahead_mask
      })
    add_attention = tf.keras.layers.add([attention1,inputs])
    attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(add_attention)

    attention2 = MultiHeadAttention(
      d_model, num_heads, name="attention_2")(inputs={
          'query': attention1,
          'key': enc_outputs,
          'value': enc_outputs,
          'mask': padding_mask
      })
    attention2 = Dropout(rate=dropout)(attention2)
    add_attention = add([attention2,attention1])
    attention2 = LayerNormalization(epsilon=1e-6)(add_attention)

    outputs = Dense(units=units, activation='relu')(attention2)
    outputs = Dense(units=d_model)(outputs)
    outputs = Dropout(rate=dropout)(outputs)
    add_attention = add([outputs,attention2])
    outputs = LayerNormalization(epsilon=1e-6)(add_attention)

    return Model(
          inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
          outputs=outputs,
          name=name)

# 测试
# sample_decoder_layer = decoder_layer(
#     units=512,
#     d_model=128,
#     num_heads=4,
#     dropout=0.3,
#     name="sample_decoder_layer")
#
# plot_model(sample_decoder_layer, to_file='decoder_layer.png', show_shapes=True)

# 定义解码器
def decoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name='decoder'):
    inputs = Input(shape=(None,), name='inputs')
    enc_outputs = Input(shape=(None, d_model), name='encoder_outputs')
    look_ahead_mask = Input(shape=(1, None, None), name='look_ahead_mask')
    padding_mask = Input(shape=(1, 1, None), name='padding_mask')

    embeddings = Embedding(vocab_size, d_model)(inputs)
    embeddings *= Lambda(lambda d_model: tf.math.sqrt( \
                        tf.cast(d_model, tf.float32)))(d_model)
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name='decoder_layer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return Model(
          inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
          outputs=outputs,
          name=name)
# 解码器测试
# sample_decoder = decoder(
#     vocab_size=21128,
#     num_layers=2,
#     units=512,
#     d_model=128,
#     num_heads=4,
#     dropout=0.3,
#     name="sample_decoder")
#
# plot_model(sample_decoder, to_file='decoder.png', show_shapes=True)

# 定义Transformer模型
def transformer(vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="transformer"):
    inputs = Input(shape=(None,), name="inputs")
    dec_inputs = Input(shape=(None,), name="dec_inputs")

    enc_padding_mask = Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='enc_padding_mask')(inputs)
    # 解码器第一个注意力块的前向掩码
    look_ahead_mask = Lambda(
      create_look_ahead_mask,
      output_shape=(1, None, None),
      name='look_ahead_mask')(dec_inputs)
    # 对编码器输出到解码器第2个注意力块的内容掩码
    dec_padding_mask = Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='dec_padding_mask')(inputs)

    enc_outputs = encoder(
          vocab_size=vocab_size,
          num_layers=num_layers,
          units=units,
          d_model=d_model,
          num_heads=num_heads,
          dropout=dropout,
        )(inputs=[inputs, enc_padding_mask])

    dec_outputs = decoder(
          vocab_size=vocab_size,
          num_layers=num_layers,
          units=units,
          d_model=d_model,
          num_heads=num_heads,
          dropout=dropout,
        )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = Dense(units=vocab_size, name="outputs")(dec_outputs)

    return Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

# 测试
# sample_transformer = transformer(
#     vocab_size=21128,
#     num_layers=4,
#     units=512,
#     d_model=128,
#     num_heads=4,
#     dropout=0.3,
#     name="sample_transformer")
#
# plot_model(sample_transformer, to_file='transformer.png', show_shapes=True)

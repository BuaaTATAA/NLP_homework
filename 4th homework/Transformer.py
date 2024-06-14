import jieba
import numpy as np
import tensorflow as tf
import keras
from pre import *
from keras import optimizers
from keras import layers
from keras import models

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='text_gen.keras',
        monitor='loss',
        save_best_only=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=3,
    ),
]


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

class TransformerEncoder(layers.Layer):
    def __init__(self, num_heads, key_dim, ff_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(key_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training):
        attn_output = self.mha(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def build_transformer_model(vocab_size, d_model, num_heads, ff_dim, maxlen):
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=d_model)
    x = embedding_layer(inputs)

    transformer_block = TransformerEncoder(num_heads=num_heads, key_dim=d_model, ff_dim=ff_dim)
    x = transformer_block(x, training=True)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(vocab_size, activation="softmax")(x)

    return models.Model(inputs=inputs, outputs=outputs)

def train(x, y, tokens, tokens_indices, epochs=200):
    x = np.asarray(x)
    y = np.asarray(y)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=4096)
    dataset = dataset.batch(128)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    maxlen = x.shape[1]
    model = build_transformer_model(vocab_size=len(tokens), d_model=256, num_heads=8, ff_dim=512, maxlen=maxlen)

    optimizer = optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    for e in range(epochs):
        model.fit(dataset, epochs=1, callbacks=callbacks_list)

        text = '青衣剑士连劈三剑，锦衫剑士一一格开。青衣剑士一声吒喝，长剑从左上角直划而下，势劲力急。锦衫剑士身手矫捷，向后跃开，避过了这剑。他左足刚着地，身子跟着弹起，刷刷两剑，向对手攻去。青衣剑士凝里不动，嘴角边微微冷笑，长剑轻摆，挡开来剑。'
        print(text, end='')
        if e % 20 == 0:
            for temperature in [0.2, 0.5, 1.0, 1.2]:
                text_cut = list(jieba.cut(text))[:60]
                print('\n temperature: ', temperature)
                print(''.join(text_cut), end='')
                for i in range(100):
                    sampled = np.zeros((1, 60))
                    for idx, token in enumerate(text_cut):
                        if token in tokens_indices:
                            sampled[0, idx] = tokens_indices[token]
                    preds = model.predict(sampled, verbose=0)[0]
                    next_index = sample(preds, temperature=temperature)
                    next_token = tokens[next_index]
                    print(next_token, end='')

                    text_cut = text_cut[1: 60] + [next_token]


if __name__ == '__main__':
    # 假设有函数 `get_single_corpus` 和 `get_dataset`
    file = DATA_PATH + '越女剑.txt'  # 修改为实际文件路径
    d = get_single_corpus(file)
    _x, _y, _tokens, _tokens_indices = get_dataset(d)
    train(_x, _y, _tokens, _tokens_indices)

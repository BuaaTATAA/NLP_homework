import jieba
import numpy as np
import tensorflow as tf
import keras
from pre import *
from keras import optimizers, layers, models, Input


callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='text_gen_seq2seq.keras',
        monitor='loss',
        save_best_only=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=1,
    ),
    keras.callbacks.EarlyStopping(
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

def build_seq2seq_model(input_dim, output_dim, latent_dim):
    encoder_inputs = Input(shape=(None,))
    x = layers.Embedding(input_dim, latent_dim)(encoder_inputs)
    encoder = layers.LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(x)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,))
    x = layers.Embedding(output_dim, latent_dim)(decoder_inputs)
    decoder_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(x, initial_state=encoder_states)
    decoder_dense = layers.Dense(output_dim, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    encoder_model = models.Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm_output, state_h, state_c = decoder_lstm(
        x, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_lstm_output)
    decoder_model = models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model

def preprocess_data(x, y, tokens_indices, maxlen=60):
    encoder_input_data = []
    decoder_input_data = []
    decoder_target_data = []

    for i in range(len(x)):
        encoder_input = np.zeros((maxlen,))
        decoder_input = np.zeros((maxlen,))
        decoder_target = np.zeros((maxlen, len(tokens_indices)))

        for t, token in enumerate(x[i][:maxlen]):
            encoder_input[t] = tokens_indices.get(token, 0)

        for t, token in enumerate(y[i][:maxlen]):
            if t > 0:
                decoder_input[t] = tokens_indices.get(y[i][t-1], 0)
            if t < len(y[i]) - 1:
                decoder_target[t, tokens_indices.get(token, 0)] = 1.0

        encoder_input_data.append(encoder_input)
        decoder_input_data.append(decoder_input)
        decoder_target_data.append(decoder_target)

    return np.array(encoder_input_data), np.array(decoder_input_data), np.array(decoder_target_data)

def train(x, y, tokens, tokens_indices, epochs=5):
    x = np.asarray(x)
    y = np.asarray(y)
    encoder_input_data, decoder_input_data, decoder_target_data = preprocess_data(x, y, tokens_indices)

    input_dim = len(tokens)
    output_dim = len(tokens)
    latent_dim = 256

    model, encoder_model, decoder_model = build_seq2seq_model(input_dim, output_dim, latent_dim)

    optimizer = optimizers.RMSprop(learning_rate=0.1)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    for e in range(epochs):
        model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                  epochs=1, callbacks=callbacks_list)

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
                    states_value = encoder_model.predict(sampled)

                    target_seq = np.zeros((1, 1))
                    target_seq[0, 0] = tokens_indices[text_cut[0]]

                    stop_condition = False
                    decoded_sentence = ''
                    while not stop_condition:
                        output_tokens, h, c = decoder_model.predict(
                            [target_seq] + states_value)

                        sampled_token_index = sample(output_tokens[0, -1, :], temperature)
                        sampled_char = tokens[sampled_token_index]
                        decoded_sentence += sampled_char

                        if sampled_char == '\n' or len(decoded_sentence) > 60:
                            stop_condition = True

                        target_seq = np.zeros((1, 1))
                        target_seq[0, 0] = sampled_token_index

                        states_value = [h, c]

                    print(decoded_sentence, end='')

if __name__ == '__main__':
    file = DATA_PATH + '越女剑.txt'
    d = get_single_corpus(file)
    _x, _y, _tokens, _tokens_indices = get_dataset(d)
    train(_x, _y, _tokens, _tokens_indices)

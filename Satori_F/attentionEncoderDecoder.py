import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

batch_size = 64
epochs = 100

input_sentences = []
output_sentences = []
output_sentences_inputs = []


def doPreprocessing(num_sentences):
    global input_sentences
    global output_sentences
    global output_sentences_inputs
    count = 0
    for line in open('pol.txt', encoding="utf-8"):
        count += 1
        if count > num_sentences:
            break
        if '\t' not in line:
            continue
        input_sentence, output = line.rstrip().split('\t')
        output_sentence = output + ' <eos>'
        output_sentence_input = '<sos> ' + output
        input_sentences.append(input_sentence)
        output_sentences.append(output_sentence)
        output_sentences_inputs.append(output_sentence_input)


def doTokenizationAndPadding():
    input_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=20000)
    input_tokenizer.fit_on_texts(input_sentences)
    input_integer_seq = input_tokenizer.texts_to_sequences(input_sentences)
    word2idx_inputs = input_tokenizer.word_index
    max_input_len = max(len(sen) for sen in input_integer_seq)

    output_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=20000, filters='')
    output_tokenizer.fit_on_texts(output_sentences + output_sentences_inputs)
    output_integer_seq = output_tokenizer.texts_to_sequences(output_sentences)
    output_input_integer_seq = output_tokenizer.texts_to_sequences(output_sentences_inputs)
    word2idx_outputs = output_tokenizer.word_index
    num_words_output = len(word2idx_outputs) + 1
    max_out_len = max(len(sen) for sen in output_integer_seq)

    encoder_input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_integer_seq, maxlen=max_input_len)
    decoder_output_sequences = tf.keras.preprocessing.sequence.pad_sequences(output_integer_seq, maxlen=max_out_len, padding='post')
    decoder_input_sequences = tf.keras.preprocessing.sequence.pad_sequences(output_input_integer_seq, maxlen=max_out_len, padding='post')
    return word2idx_inputs, max_input_len, word2idx_outputs, num_words_output, max_out_len, encoder_input_sequences, \
        decoder_output_sequences, decoder_input_sequences


def doEmbedding():
    embeddings_dictionary = dict()
    glove_file = open('glove.6B.100d.txt', encoding="utf8")
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()
    num_words = min(20000, len(word2idx_inputs) + 1)
    embedding_matrix = np.zeros((num_words, 100))
    for word, index in word2idx_inputs.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    embedding_layer = tf.keras.layers.Embedding(num_words, 100, weights=[embedding_matrix], input_length=max_input_len)
    return embedding_layer


def createAndTrainModel():
    decoder_targets_one_hot = np.zeros((len(input_sentences), max_out_len, num_words_output), dtype='float32')
    for i, d in enumerate(decoder_output_sequences):
        for t, word in enumerate(d):
            decoder_targets_one_hot[i, t, word] = 1
    encoder_inputs = tf.keras.layers.Input(shape=(max_input_len,))
    x = embedding_layer(encoder_inputs)
    encoder = tf.keras.layers.LSTM(256, return_state=True)
    encoder_outputs, h, c = encoder(x)
    encoder_states = [h, c]
    decoder_inputs_placeholder = tf.keras.layers.Input(shape=(max_out_len,))
    decoder_embedding = tf.keras.layers.Embedding(num_words_output, 256)
    decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)
    decoder_lstm = tf.keras.layers.LSTM(256, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x, initial_state=encoder_states)
    decoder_attention = tf.keras.layers.AdditiveAttention()
    decoder_outputs += decoder_attention([decoder_outputs, decoder_outputs, decoder_outputs])
    decoder_dense = tf.keras.layers.Dense(num_words_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = tf.keras.models.Model([encoder_inputs, decoder_inputs_placeholder], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([encoder_input_sequences, decoder_input_sequences], decoder_targets_one_hot, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    return encoder_inputs, encoder_states, decoder_embedding, decoder_lstm, decoder_dense


def preparePrediction():
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
    decoder_state_input_h = tf.keras.layers.Input(shape=(256,))
    decoder_state_input_c = tf.keras.layers.Input(shape=(256,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_inputs_single = tf.keras.layers.Input(shape=(1,))
    decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)
    decoder_outputs, h, c = decoder_lstm(decoder_inputs_single_x, initial_state=decoder_states_inputs)
    decoder_states = [h, c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model([decoder_inputs_single] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    return encoder_model, decoder_model


def translate_sentence(_input_seq):
    states_value = encoder_model.predict(_input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx_outputs['<sos>']
    eos = word2idx_outputs['<eos>']
    _output_sentence = []

    for _ in range(max_out_len):
        output_tokens, _h, _c = decoder_model.predict([target_seq] + states_value)
        idx = np.argmax(output_tokens[0, 0, :])
        if eos == idx:
            break
        _word = ''
        if idx > 0:
            _word = idx2word_target[idx]
            _output_sentence.append(_word)
        target_seq[0, 0] = idx
        states_value = [_h, _c]
    return ' '.join(_output_sentence)


doPreprocessing(5000)
word2idx_inputs, max_input_len, word2idx_outputs, num_words_output, max_out_len, encoder_input_sequences, \
    decoder_output_sequences, decoder_input_sequences = doTokenizationAndPadding()
embedding_layer = doEmbedding()
encoder_inputs, encoder_states, decoder_embedding, decoder_lstm, decoder_dense = createAndTrainModel()
encoder_model, decoder_model = preparePrediction()
idx2word_input = {v: k for k, v in word2idx_inputs.items()}
idx2word_target = {v: k for k, v in word2idx_outputs.items()}

bleu = []
for _ in range(5000):
    i = np.random.choice(len(input_sentences))
    input_seq = encoder_input_sequences[i:i + 1]
    translation = translate_sentence(input_seq)
    print('-')
    print('Input:', input_sentences[i])
    print('Response:', translation)
    bleu.append(sentence_bleu([output_sentences[i]], translation))
print(np.mean(bleu))

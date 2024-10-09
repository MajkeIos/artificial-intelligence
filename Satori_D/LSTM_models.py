import tensorflow as tf
import numpy as np

txt_fie = "alice.txt"

data = open(txt_fie).read()
print(f'Number of characters = {len(data)}')

corpus = data.lower().split("\n")

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
print(f'Number of different words = {total_words}')

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
print(f'Max sequence length = {max_sequence_len}')
input_sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(input_sequences,
                                                                         maxlen=max_sequence_len, padding='pre'))

predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
label = tf.keras.utils.to_categorical(label, num_classes=total_words)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_len - 1))
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150, return_sequences=True)))
# model.add(tf.keras.layers.Dropout(0.2))
# model.add(tf.keras.layers.LSTM(100))
# model.add(tf.keras.layers.Dense(total_words / 2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
# model.add(tf.keras.layers.Dense(total_words, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())

model = tf.keras.models.load_model("Alice_LSTM")
model.evaluate(predictors, label)

model = tf.keras.models.load_model("alice_LSTM/20")
model.evaluate(predictors, label)

seed_text = "alice was"
next_words = 20

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word

print(seed_text)

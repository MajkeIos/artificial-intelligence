import random
import math
import numpy as np

n = 3

data = open('alice.txt').read()

sentences = data.lower().split("\n")
all_words = []
for sentence in sentences:
    for word in sentence.split(" "):
        if word != '':
            all_words.append(word)

n_grams = {}

for i in range(n - 1, len(all_words)):
    context = ""
    for j in range(n - 1, 0, -1):
        context += all_words[i - j] + " "
    if context not in n_grams:
        n_grams[context] = []
    n_grams[context].append(all_words[i])

seed_text = "Alice was beginning to get very tired of sitting"
next_words = 20

words = seed_text.lower().split(" ")

for i in range(0, next_words):
    context = ""
    for j in range(n - 1, 0, -1):
        context += words[len(words) - j] + " "
    next_word = random.choice(list(n_grams[context]))
    words.append(next_word)

print(" ".join(words))

sigma = 0
cur_pos = len(seed_text.split(" "))

for i in range(0, next_words):
    context = ""
    for j in range(n - 1, 0, -1):
        context += words[cur_pos - j] + " "
    possible_words = n_grams[context]
    chosen_word = words[cur_pos]
    chosen_word_counter = 0
    for word in possible_words:
        if word == chosen_word:
            chosen_word_counter += 1
    sigma += math.log(float(chosen_word_counter)/float(len(possible_words)))
    cur_pos += 1

e_sigma = np.exp(sigma / next_words)

print(1.0 / e_sigma)

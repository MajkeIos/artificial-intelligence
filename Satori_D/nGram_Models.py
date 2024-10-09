import random
import math
import numpy as np

n = 3

data = open('Alice.txt').read()

data = data.lower().split(" ")
all_words = []
for i in range(0, len(data)):
    for word in data[i].split("\n"):
        valids = []
        for character in word:
            if character.isalpha():
                valids.append(character)
        if len(valids) > 0:
            all_words.append(''.join(valids))

n_grams = {}

for i in range(n - 1, len(all_words)):
    context = ""
    for j in range(n - 1, 0, -1):
        context += all_words[i - j] + " "
    if context not in n_grams:
        n_grams[context] = []
    n_grams[context].append(all_words[i])

seed_text = "Alice was"
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

for i in range(0, len(all_words) - n + 1):
    context = ""
    for j in range(n - 1, 0, -1):
        context += all_words[cur_pos - j] + " "
    possible_words = n_grams[context]
    chosen_word = all_words[cur_pos]
    chosen_word_counter = 0
    for word in possible_words:
        if word == chosen_word:
            chosen_word_counter += 1
    sigma += math.log(float(chosen_word_counter)/float(len(possible_words)))
    cur_pos += 1

e_sigma = np.exp(sigma / (len(all_words) - n + 1))

print("Perplexity: " + str(1.0 / e_sigma))

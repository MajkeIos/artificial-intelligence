import glob
import os
import random
import sys

def prepare_words():
    global common_dict
    global satisfying_words
    for i, dir in enumerate(dirs):
        for j, file_path in enumerate(glob.glob(os.getcwd() + '/' + dir + '/*')):
            with open(file_path) as file:
                review = file.read().split()
                for word in review:
                    if word in dicts[i].keys():
                        dicts[i][word] += 1
                    else:
                        dicts[i][word] = 1
                    if word in common_dict.keys():
                        common_dict[word] += 1
                    else:
                        common_dict[word] = 1
    common_dict = dict(sorted(common_dict.items(), key=lambda item: item[1], reverse=True))
    for i, word in enumerate(common_dict.keys()):
        if i == words_to_choose:
            break
        satisfying_words[word] = i

def predicate(x, w):
    result = 0
    for i in range(len(w)):
        result += x[i] * w[i]
    return 1 if result >= 0 else 0

def train():
    features = [[], []]
    counter = 0
    for i, dir in enumerate(dirs):
        for j, file_path in enumerate(glob.glob(os.getcwd() + '/' + dir + '/*')):
            temp = [0] * words_to_choose
            with open(file_path) as file:
                review = file.read().split()
                for word in review:
                    if word in satisfying_words.keys():
                        temp[satisfying_words[word]] = 1
            features[i].append(temp)
    for i in range(words_to_choose):
        weight_vect.append(random.random())
    for i in range(epoch):
        folder = random.randint(0, 1)
        file = random.randint(0, 699)
        x = features[folder][file]
        y_prim = predicate(x, weight_vect)
        if folder == 1 and y_prim == 0:
            for i in range(words_to_choose):
                weight_vect[i] += x[i] * learning_rate
        elif folder == 0 and y_prim == 1:
            for i in range(words_to_choose):
                weight_vect[i] -= x[i] * learning_rate

def test():
    test_dir = sys.argv[3]
    for file_path in glob.glob(os.getcwd() + '/' + test_dir + '/*'):
        features = [0] * words_to_choose
        with open(file_path) as file:
            review = file.read().split()
            for word, i in satisfying_words.items():
                if word in review:
                    features[i] = 1
        print(f'{os.path.basename(file_path)} {predicate(features, weight_vect) * 2 - 1}')
                    

dirs = [sys.argv[2], sys.argv[1]]
dicts = [{}, {}]
common_dict = {}
satisfying_words = {}
words_to_choose = 1000
epoch = 250000
learning_rate = 0.2

weight_vect = []

prepare_words()
train()
test()

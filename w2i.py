import json
import pickle

words = []
idx = 0
word2idx = {}
with open('data/glove.6B.50d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
# print(word2idx)
# pickle.dump(word2idx, open( "data/embed_map.p", "wb" ))

# x = open('temp.txt', 'w', encoding='utf-8')
# x.write(str(word2idx))
# x.close()

# if "us" in word2idx:
#     print(word2idx["us"])
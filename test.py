import pickle

from com.train_utils import load_t_d

train_data, dev_data = load_t_d()
for line in train_data[:3]:
    print(line)

w2id=pickle.load(open("data/word2id.obj","rb"))
max = 0
for k in w2id.keys():
    if w2id[k] == 0:
        print(k)
    if w2id[k] > max:
        max =w2id[k]
print(max) # 96972
print(w2id["。"])
print(w2id["！"])
print(w2id["？"])
print(w2id[";"])
print(w2id["；"])
print(w2id["!"])
print(w2id["?"])
print(len(list(w2id.keys())))
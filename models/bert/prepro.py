import pickle

from com.utils import load_t_d

def get_sentence_pairs(dt_set,min_len=5,max_len=350):
    """
    将数据集中的句子从中间切成一对句子
    :param dt_set: train/test/dev
    :return: list of sentence sequence
    """
    sentence_pair_list=[]
    size=len(dt_set)
    print("converting data...")
    # print("size:",size)
    for i in range(size):
        # print("{} in {}".format(i,size))
        line = dt_set[i]
        para = []
        # [s for s in line if type(s) == list and len(s) > min_len]
        for s in line:
            if type(s) == list and len(s) > min_len:
                if len(s) > max_len:
                    s = s[:max_len]
                    para.append(s)
                else:
                    para.append(s)
        for p in para:
            mid_pos = len(p)//2
            sentence_pair_list.append([p[:mid_pos],p[mid_pos:]])  # 将句对以列表形式加入pair_list中
    return sentence_pair_list

def update_w2id(w2id_path="../../data/word2id.obj"):
    w2id = pickle.load(open(w2id_path, "rb"))
    w2id["<PAD>"] = 0
    w2id["<MASK>"] = 96973  # 原本w2id的最大id
    w2id["<SEP>"] = 96974
    w2id["<CLS>"] = 96975
    pickle.dump(w2id, open(w2id_path, "wb"))

def load_sentence_pair(dt_path="../../data/",max_len=350):
    train_data, dev_data = load_t_d(dt_path=dt_path)
    # train
    train_pair = get_sentence_pairs(train_data,max_len=max_len)
    # dev
    dev_pair = get_sentence_pairs(dev_data,max_len=max_len)
    # testa
    with open(dt_path + 'testa.pickle', 'rb') as f: #FIXME: 等testb开发后可加上
        testa_data = pickle.load(f)
    teata_data = sorted(testa_data, key=lambda x: len(x[1]))
    testa_pair = get_sentence_pairs(teata_data,max_len=max_len)
    return train_pair + dev_pair + testa_pair

if __name__ == '__main__':
    # update_w2id() # w2id size from 96973 tp 96976
    sentence_pairs=load_sentence_pair()
    # print(len(sentence_pairs))



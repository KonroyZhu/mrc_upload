import argparse
import pickle

import numpy as np
import torch

from com.preprocess import transform_data_to_id


def pad_answer(batch):
    output = []
    length_info = [len(x[0]) for x in batch]
    max_length = max(length_info)
    for one in batch:
        output.append([x + [0] * (max_length - len(x)) for x in one])
    return output


def get_model_parameters(model):
    total = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            tmp = 1
            for a in parameter.size():
                tmp *= a
            total += tmp
    return total


def padding(sequence, pads=0, max_len=None,limit_max=True, dtype='int32', return_matrix_for_size=False):
    # we should judge the rank
    if True or isinstance(sequence[0], list):
        v_length = [len(x) for x in sequence]  # every sequence length
        seq_max_len = max(v_length)
        if limit_max:
            if (max_len is None) or (max_len > seq_max_len):
                max_len = seq_max_len
        v_length = list(map(lambda z: z if z <= max_len else max_len, v_length))
        x = (np.ones((len(sequence), max_len)) * pads).astype(dtype)
        for idx, s in enumerate(sequence):
            trunc = s[:max_len]
            x[idx, :len(trunc)] = trunc
        if return_matrix_for_size:
            v_matrix = np.asanyarray([map(lambda item: 1 if item < line else 0, range(max_len)) for line in v_length],
                                     dtype=dtype)
            return x, v_matrix
        return x, np.asarray(v_length, dtype='int32')
    else:
        seq_len = len(sequence)
        if max_len is None:
            max_len = seq_len
        v_vector = sequence + [0] * (max_len - seq_len)
        padded_vector = np.asarray(v_vector, dtype=dtype)
        v_index = [1] * seq_len + [0] * (max_len - seq_len)
        padded_index = np.asanyarray(v_index, dtype=dtype)
        return padded_vector, padded_index


def shuffle_data(data, axis=1):
    pool = {}
    for one in data:
        length = len(one[axis])
        if length not in pool:
            pool[length] = []
        pool[length].append(one)
    for one in pool:
        np.random.shuffle(pool[one])
    length_lst = list(pool.keys())
    np.random.shuffle(length_lst)
    return [x for y in length_lst for x in pool[y]]


def load_data(data_path="../data/testa_seg.pkl", word2id_path='../data/word2id.obj'):
    with open(word2id_path, 'rb') as f:
        word2id = pickle.load(f)
    raw_data = pickle.load(open(data_path, "rb"))
    transformed_data = transform_data_to_id(raw_data, word2id)
    data = [x + [y[2]] for x, y in zip(transformed_data, raw_data)]
    data = sorted(data, key=lambda x: len(x[1]))
    print('test data size {:d}'.format(len(data)))
    return data


def load_model(model_path="../net/mwan_f1.pt", cuda=False):
    with open(model_path, 'rb') as f:
        model = torch.load(f)
    if cuda:
        model.cuda()
    return model


def pad_wrong_answer(answer_list):
    # 3680
    # 7136
    # 这两批数据中有alternative answer长度小于3的数据，需要补齐否则无法处理
    # 该方法通过复制ans[0]补齐数据
    padded_list = []
    for ans in answer_list:
        ll = len(ans)
        if not ll == 3:
            for _ in range(3 - ll):
                ans += [ans[0]]
        padded_list.append(ans)
    padded_list = pad_answer(padded_list)
    return padded_list


def to_onehot(x, n_class=3):
    values = np.array(x)
    b=np.eye(n_class)[values]
    return b

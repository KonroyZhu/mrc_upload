import pickle
import random

import numpy as np
import copy
from models.bert.prepro import load_sentence_pair


class Data_utils:
    def __init__(self, sentence_pairs, w2id):
        self.max_len = 400
        self.sentence_pairs = sentence_pairs
        self.w2id = w2id
        self.vocab_size = len(list(w2id.keys()))
        self.num_pairs = len(sentence_pairs)

    def __getitem__(self, index):
        """
        :param index: self.sentence_pairs中的句子下标
        :return: 该下标句子对应的bert_input,bert_label,segment_label
        """
        t1, t2, is_next_label = self.random_sent(index)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        t1 = [self.w2id["<CLS>"]] + t1_random + [self.w2id["<SEP>"]]
        t2 = t2_random + [self.w2id["<SEP>"]]

        t1_label = [self.w2id["<PAD>"]] + t1_label + [self.w2id["<PAD>"]]  # 只取t1_label(真实的词语)其余的用pad填补
        t2_label = t2_label + [self.w2id["<PAD>"]]

        # segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])  # 将padding留给com.utils 中的工具
        # bert_input = (t1 + t2)[:self.seq_len]
        bert_input = (t1 + t2)
        # bert_label = (t1_label + t2_label)[:self.seq_len]
        mask_label = (t1_label + t2_label)
        return bert_input, mask_label, segment_label, is_next_label

    def random_word(self, sentence):
        # tokens=sentence FIXME:!!python中的复制为物理地址的传递，即对tokens的操作会作用在sentence上，应换成浅复制(如下
        tokens = copy.copy(sentence)
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    # tokens[i] = vocab.mask_index
                    tokens[i] = self.w2id["<MASK>"]

                # 10% randomly change token to random token
                elif prob < 0.9:
                    # tokens[i] = random.randrange(len(vocab))
                    tokens[i] = random.randrange(96972)  # vocab len
                # 10% randomly change token to current token
                else:
                    # tokens[i] = vocab.stoi.get(token, vocab.unk_index)
                    tokens[i] = sentence[i]  # 原词语的id

                # output_label.append(vocab.stoi.get(token, vocab.unk_index))
                output_label.append(sentence[i])  # 原词语的id TODO：to be verified

            else:
                # tokens[i] = vocab.stoi.get(token, vocab.unk_index)
                tokens[i] = sentence[i]
                output_label.append(0)

        return tokens, output_label

    def random_sent(self, index):
        t1, t2 = self.sentence_pairs[index][0], self.sentence_pairs[index][1]

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            rand_index = np.random.random_integers(0, self.num_pairs - 1)
            rand_index2 = np.random.random_integers(0, 1)
            return t1, self.sentence_pairs[rand_index][rand_index2], 0


if __name__ == '__main__':
    w2id = pickle.load(open("../../data/word2id.obj", "rb"))
    sentence_pairs = load_sentence_pair()
    dt_u = Data_utils(sentence_pairs, w2id)
    inp1, t_label1, seg_label1, is_next_label1 = dt_u.__getitem__(0)
    inp2, t_label2, seg_label2, is_next_label2 = dt_u.__getitem__(0)
    inp3, t_label3, seg_label3, is_next_label3 = dt_u.__getitem__(0)
    print(inp3)
    print(t_label3)
    print(seg_label3)
    print(is_next_label3)

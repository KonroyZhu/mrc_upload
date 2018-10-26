import numpy as np
import torch
import torch.nn as nn

from models.bert.bert import BERT


class BERTLM(nn.Module):
    def __init__(self, bert: BERT, vocab_size):
        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)


class NextSentencePrediction(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        :param x: (b,t,h)
        :return: (b,2)
        """
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(nn.Module):

    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        :param x: (b,t,h)
        :return: (b,t,vocab)
        """
        return self.softmax(self.linear(x))


if __name__ == '__main__':
    vocab_size = 1000
    x = torch.LongTensor(np.random.randint(0, vocab_size, size=(32, 50)))
    seg_info = torch.LongTensor(np.random.randint(0, 3, size=(32, 50)))
    bert = BERT(vocab_size=vocab_size)
    bert_lm = BERTLM(vocab_size=vocab_size, bert=bert)
    print("BERT output:", bert(x, seg_info).shape)
    next_s, mask_lm = bert_lm(x, seg_info)
    print("next sentence:", next_s.shape)  # (b,2)
    print("mask lm:", mask_lm.shape)  # (b,t,vocab)

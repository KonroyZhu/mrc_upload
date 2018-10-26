import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.bert.transformer import Transformer


class BERT(nn.Module):
    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.feed_forward_hidden = hidden * 4

        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [Transformer(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info):
        """
        :param x: (b,t)
        :param segment_info: (b,t)
        :return: (b,t,h)
        """
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)  # (b,1,t,t)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x)

        return x


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)  # (max_len,1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()  # (d_model/2)

        pe[:, 0::2] = torch.sin(position * div_term)  # (max_len,d_model/2) from 0 to the end step =2
        pe[:, 1::2] = torch.cos(position * div_term)  # (max_len,d_model/2) from 1 to the end step =2

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: (b,t)
        :return: (b,t,h)
        """
        return self.pe[:, :x.size(1)]


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout=0.1):
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label):
        """
        :param sequence: (b,t)
        :param segment_label: (b,t)
        :return: (b,t,h)
        """
        token = self.token(sequence)
        position = self.position(sequence)
        segment = self.segment(segment_label)
        x = token + position + segment
        return self.dropout(x)


if __name__ == '__main__':
    vocab_size = 1000
    x = torch.LongTensor(np.random.randint(0, vocab_size, size=(32, 50)))
    seg_info = torch.LongTensor(np.random.randint(0, 3, size=(32, 50)))
    bert = BERT(vocab_size=vocab_size)
    print(bert(x, seg_info).shape)

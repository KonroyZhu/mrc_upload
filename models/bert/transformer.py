import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    def __init__(self, hidden, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden))
        self.beta = nn.Parameter(torch.zeros(hidden))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)  # 均值
        std = x.std(-1, keepdim=True)  # 方差
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class Transformer(nn.Module):
    def __init__(self, hidden, num_head, feed_forward_hidden, dropout):
        super().__init__()
        self.multi_head_att = MultiHeadAttention(num_head=num_head, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_layer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        :param x: (b,t,h)
        :return: (b,t,h)
        """
        x = self.input_sublayer(x, lambda _x: self.multi_head_att.forward(_x, _x, _x))
        x = self.output_layer(x, self.feed_forward)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, d_model, dropout=0.1):
        super().__init__()
        assert d_model % num_head == 0
        self.d_k = d_model // num_head
        self.num_head = num_head

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        """
        :param query: (b,q,h)
        :param key: (b,k,h)
        :param value: (b,v,h)
        :return: (b.q.h)
        """
        batch_size = query.size(0)
        assert value.size(1) == key.size(1)  # value should be equal with key
        # linear projection
        q = self.q_linear(query).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)  # (b,nh,q,h/nh)
        k = self.k_linear(key).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)  # (b,nh,k,h/nh)
        v = self.q_linear(value).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)  # (b,nh,v,h/nh)

        # attention
        score = torch.matmul(q, k.transpose(2, 3) / math.sqrt(q.size(-1)))  # (b,nh,q,k)
        k_attn = F.softmax(score, dim=-1)  # (b,nh,q,k)
        k_attn = self.dropout(k_attn)
        x = torch.matmul(k_attn, v)  # (b,nh.q.k) (b,nh,v,h/nh) note that v == k -> (b,nh,q,h/nh)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_head * self.d_k)  # (b.q,h)

        return self.out_linear(x)  # (b.q.h)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def activate(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def forward(self, x):
        """
       :param x: (b,t,h)
       :return: (b,t,h)
       """
        return self.W2(self.dropout(self.activate(self.W1(x))))


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        :param x: (b,x,h)
        :param sublayer: nn.Module
        :return: (b,x,h)
        """
        return x + self.dropout(sublayer(self.norm(x)))


if __name__ == '__main__':
    x = torch.FloatTensor(32, 50, 128)
    transformer = Transformer(128, 4, 100, 0.2)
    print(transformer(x).shape)

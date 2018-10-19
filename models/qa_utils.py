import math
import pickle
import time
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from com.utils import padding
from models.pred_layer import Pred_Layer


class Highway(nn.Module):  # 1505.00387.pdf
    def __init__(self, layer_num: int, hidden_size: int):
        super().__init__()
        self.num = layer_num
        # 输入输出的隐层维度一致
        self.linear = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(self.num)])
        self.gate = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(self.num)])

    def forward(self, x):
        # x: (b,t,h) highway 中没有使用conv->无需将h放在中间作为in_channel
        for i in range(self.num):
            T = torch.sigmoid(self.gate[i](x))  # T as the transform gate (b,x,h) (h,h)
            H = F.relu(self.linear[i](x))  # H is an affine transform follow by activation
            C = (1 - T)  # (b,x,h)
            x = H * T + C * x  # (b,x,h)
        return x


class Emb_Wrapper(nn.Module):
    def __init__(self, layer_num: int, emb_dim: int, hidden_size: int):
        super().__init__()
        self.conv_projector = nn.Conv1d(in_channels=emb_dim, out_channels=hidden_size, padding=5 // 2, kernel_size=5)
        self.highway = Highway(layer_num, hidden_size)

    def forward(self, x):
        # x: (b,t,h)
        _x = self.conv_projector(x.transpose(1, 2)).transpose(1, 2)
        output = self.highway(_x)
        return output


class PosEncoder(nn.Module):
    def __init__(self, d_model, dropout, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()  # (max_length,h)
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)  # (max_length,)
        div_term = (torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).float().exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1,max_length,h)
        self.register_buffer('pe', pe)  # 添加buffer可以根据实际情况截取max_length

    def forward(self, x):
        # x: (b,t,h)
        x = x + self.pe[:, :x.size(1)]  # (b,t,h) + (1,t,h) 自动广播
        return self.dropout(x)


class SelfAttention(nn.Module):
    def __init__(self, encoder_size: int, num_header: int):
        super().__init__()
        self.encoder_size = encoder_size
        self.num_header = num_header
        Wo = torch.empty(encoder_size, encoder_size)
        Wqs = [torch.empty(encoder_size, int(encoder_size / num_header)) for _ in range(num_header)]
        Wks = [torch.empty(encoder_size, int(encoder_size / num_header)) for _ in range(num_header)]
        Wvs = [torch.empty(encoder_size, int(encoder_size / num_header)) for _ in range(num_header)]
        # init
        nn.init.kaiming_uniform_(Wo)
        for i in range(num_header):
            nn.init.xavier_uniform_(Wqs[i])
            nn.init.xavier_uniform_(Wvs[i])
            nn.init.xavier_uniform_(Wks[i])
        self.Wo = nn.Parameter(Wo)
        self.Wqs = nn.ParameterList([nn.Parameter(X) for X in Wqs])
        self.Wks = nn.ParameterList([nn.Parameter(X) for X in Wks])
        self.Wvs = nn.ParameterList([nn.Parameter(X) for X in Wvs])

    def forward(self, x):
        # x: (b,h,t)
        WQs, WKs, WVs = [], [], []
        dk = self.encoder_size / self.num_header
        sqrt_dk_inv = 1 / math.sqrt(dk)
        x = x.transpose(1, 2)  # (b,t,h)
        for i in range(self.num_header):
            WQs.append(torch.matmul(x, self.Wqs[1]))  # (b.t.h) (h,h/nh)
            WKs.append(torch.matmul(x, self.Wks[1]))
            WVs.append(torch.matmul(x, self.Wvs[1]))
        heads = []
        for i in range(self.num_header):
            out = torch.bmm(WQs[i], WKs[i].transpose(1, 2))  # 分母 公式(1) attention is all you need
            out = torch.mul(out, sqrt_dk_inv)  # *(1/分子) 公式(1) attention is all you need
            out = F.softmax(out, dim=2)
            headi = torch.bmm(out, WVs[i])  # 公式(1) softmax后乘V
            heads.append(headi)
        head = torch.cat(heads, dim=2)
        out = torch.matmul(head, self.Wo)
        return out.transpose(1, 2)


class LayerNorm_Buffer(nn.Module):
    def __init__(self, in_ch, max_length=512):
        super().__init__()
        normalized_shape = [max_length, max_length]
        weight = nn.Parameter(torch.Tensor(*normalized_shape))
        bias = nn.Parameter(torch.Tensor(*normalized_shape))
        # weight.data.fill_(1) Fixme: 取1初始化权重导致nan !!!
        initrange=0.1
        nn.init.uniform_(weight, -initrange, initrange)
        bias.data.zero_()
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

    def forward(self, x):
        # x: (b,h,t)
        norm_shape = [x.size(1), x.size(2)]
        norm = nn.LayerNorm(norm_shape, elementwise_affine=False)
        output = norm(x)
        w = self.weight[:x.size(2), :x.size(2)]  # (t,t)
        b = self.bias[0, :x.size(2)]  # (t)
        # print("norm weight",w) # TODO: 跑起后观察w是否有被优化
        output = torch.add(torch.matmul(output, w), b)
        return output


class Encoder_Block(nn.Module):
    def __init__(self, conv_num: int, in_channels: int, k: int, num_header=2, dropout=0.1):
        super().__init__()
        # in_ch 相当于输入的encoder_size
        self.dropout = dropout
        self.num_conv = conv_num
        self.position_encoder = PosEncoder(dropout=dropout, d_model=in_channels)
        self.norm_begin = LayerNorm_Buffer(in_channels)
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=in_channels, out_channels=in_channels, padding=k // 2, kernel_size=k)
             for _ in range(conv_num)])  # t + 2*pad//2 - (k-1) (pad=k//2保证输出一致)
        self.norms = nn.ModuleList([LayerNorm_Buffer(in_channels)
                                    for _ in range(conv_num)])
        self.self_att = SelfAttention(in_channels, num_header)
        self.norm_end = LayerNorm_Buffer(in_channels)
        self.full_connect = nn.Linear(in_channels, in_channels, bias=True)

    def forward(self, x):
        # x (b,h,x)
        out = self.position_encoder(x.transpose(1, 2)).transpose(1, 2)  # (b,h,t)
        res = self.norm_begin(out)
        for i, conv in enumerate(self.convs):
            out = conv(out)  # 两层conv后输出形状一致
            out = F.relu(out)
            # #print("out",out.shape)
            # #print("res",res.shape)
            out = out + res
            if (i + 1) % 2 == 0:
                p_drop = self.dropout * (i + 1) / self.num_conv
                out = F.dropout(out, p_drop)
            res = out
            out = self.norms[i](out)
        out = self.self_att(out)
        out = out + res
        out = F.dropout(out, self.dropout)
        res = out
        out = self.norm_end(out)
        out = self.full_connect(out.transpose(1, 2)).transpose(1, 2)
        out = F.relu(out)
        out = out + res
        out = F.dropout(out, self.dropout)
        return out


class CQAttention(nn.Module):
    def __init__(self, encoder_size, dropout=0.1):
        super().__init__()
        self.Wo = nn.Linear(3 * encoder_size, 1, bias=True)
        self.dropout = dropout
        nn.init.uniform_(self.Wo.weight, -math.sqrt(1 / encoder_size), math.sqrt(1 / encoder_size))

    def forward(self, C, Q):
        # C: (b,p,h)
        # Qk: (b,q,h)
        ss = []

        Ct = C.unsqueeze(2)  # (b,p,1,h)
        Qt = Q.unsqueeze(1)  # (b,1,q,h)

        CQ = torch.mul(Ct, Qt)  # (b,p,q,h)
        shape = (C.size(0), C.size(1), Q.size(1), C.size(2))  # (b,p,q,h)
        S = torch.cat([Ct.repeat(1, 1, Q.size(1), 1), Qt.repeat(1, C.size(1), 1, 1), CQ], dim=3)  # (b,p,q,3h)
        S = self.Wo(S).view(S.size(0), S.size(1), -1)  # (b,p,q)
        Sq = F.softmax(S, dim=2)  # q attention
        Sc = F.softmax(S, dim=1)  # p attention
        A = torch.bmm(Sq, Q)  # (b,p,q) (b,q,h)-> (b,p,h)
        B = torch.bmm(torch.bmm(Sq, Sc.transpose(1, 2)), C)  # (b,p,q) (b,q,p) (b,p,h) -> (b,p,h)

        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)  # (b,p,4h)
        out = F.dropout(out, self.dropout)
        return out


"""
class QA_Net(nn.Module):  # param:
    def __init__(self, options, embedding=None):
        super(QA_Net, self).__init__()
        self.drop_out = options["dropout"]
        self.opts = options
        vocab_size = self.opts["vocab_size"]
        encoder_size = self.opts["hidden_size"]
        if embedding is None:
            embedding_size = options["emb_size"]
            self.embedding = nn.Embedding(vocab_size + 1, embedding_dim=embedding_size)
            initrange = 0.1
            nn.init.uniform_(self.embedding.weight, -initrange, initrange)  # embedding初始化为-0.1~0.1之间
            # #print("embedding initialized")
        else:
            embedding_size = np.shape(embedding)[1]  # (vocab_size,embedding_dim)
            self.embedding = nn.Embedding(vocab_size + 1, embedding_dim=embedding_size)
            self.embedding.from_pretrained(embedding, freeze=False)  # TODO:斟酌一下要不要freeze
        self.emb_size = embedding_size
        self.q_conv_project = nn.Conv1d(in_channels=embedding_size, out_channels=encoder_size, kernel_size=1)
        self.p_conv_project = nn.Conv1d(in_channels=embedding_size, out_channels=encoder_size, kernel_size=1)
        self.a_conv_project = nn.Conv1d(in_channels=embedding_size, out_channels=embedding_size, kernel_size=1)

        # self.a_conv_project = nn.Conv1d(in_channels=embedding_size, out_channels=encoder_size, kernel_size=1)
        self.q_highway = Highway(layer_num=2, hidden_size=encoder_size)
        self.p_highway = Highway(layer_num=2, hidden_size=encoder_size)
        self.a_highway = Highway(layer_num=2, hidden_size=embedding_size)

        self.p_encoder = Encoder_Block(conv_num=4, in_channels=encoder_size, k=7)
        self.q_encoder = Encoder_Block(conv_num=4, in_channels=encoder_size, k=7)
        self.a_encoder = Encoder_Block(conv_num=4, in_channels=embedding_size, k=7)

        # answer transform
        self.a_attention = nn.Linear(embedding_size, 1, bias=False)

        # Context-Query Attention
        self.cq_att = CQAttention(encoder_size)
        self.x_conv_project = nn.Conv1d(in_channels=4 * encoder_size, out_channels=encoder_size, padding=5 // 2,
                                        kernel_size=5)

        encoder_block = Encoder_Block(conv_num=2, in_channels=encoder_size, k=5)
        self.model_enc_blks = nn.ModuleList([encoder_block] * 7)  # 7 个 encoder block

        # RESIZE
        self.q_encoder_resize = nn.Conv1d(in_channels=encoder_size, out_channels=2 * encoder_size, kernel_size=5)
        self.M2_resize = nn.Conv1d(in_channels=encoder_size, out_channels=2 * encoder_size, kernel_size=5)
        self.predictio_layer = Pred_Layer(options)

        self.initiation()

    def initiation(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):  # 用0.1来限制，初始化所有nn.Linear的权重
                print("initializing Linear:", module)
                nn.init.xavier_uniform_(module.weight, 0.1)

    def forward(self, inputs):
        [query, passage, answer, ids, is_train, is_argmax] = inputs
        # Embedding
        q_embedding = self.embedding(query)
        p_embedding = self.embedding(passage)
        a_embeddings = self.embedding(answer)
        a_embeddings = a_embeddings.view(-1, answer.size(1), self.emb_size)  # (3b,a,h)

        print("p", p_embedding)
        print("q", q_embedding)
        print("a", a_embeddings)

        q_conv_projection = self.q_conv_project(q_embedding.transpose(1, 2)).transpose(1, 2)  # (b,q,emb)-> (b,q,h)
        p_conv_projection = self.p_conv_project(p_embedding.transpose(1, 2)).transpose(1, 2)  # (b,q,emb)-> (b,p,h)
        a_conv_projection = self.a_conv_project(a_embeddings.transpose(1, 2)).transpose(1,
                                                                                        2)  # (b,q,emb)-> (3b,a,emb) FIXME

        # # two-layer highway network
        q_highway = self.q_highway(q_conv_projection)  # (b,q,h)
        p_highway = self.p_highway(p_conv_projection)  # (b,p,h)
        a_highway = self.a_highway(a_conv_projection)  # (3b,a,emb)

        p_encoder = self.p_encoder(p_highway.transpose(1, 2)).transpose(1, 2)
        q_encoder = self.q_encoder(q_highway.transpose(1, 2)).transpose(1, 2)
        a_encoder = self.a_encoder(a_highway.transpose(1, 2)).transpose(1, 2)

        # #print("p/q_encoder: {}".format(p_encoder.shape))
        # #print("a_encoder: {}".format(a_encoder.shape))
        # a score
        a_score = F.softmax(self.a_attention(a_encoder), 1)  # (3b,a,1)
        a_output = a_score.transpose(2, 1).bmm(a_encoder).squeeze()  # (3b,1,a) bmm (3b,a,h)-> (3b,1,h)
        print(a_output.shape)
        a_embedding = a_output.view(answer.size(0), 3, a_encoder.size(2))  # (b,3,h)

        X = self.cq_att(p_encoder, q_encoder)
        # print("CQ(X): {}".format(X)) # (b.q.4h)

        M1 = self.x_conv_project(X.transpose(1, 2))  # (b,h,q)
        for enc in self.model_enc_blks:  # 7个
            M1 = enc(M1)
        M2 = M1
        # FIXME: q_encoder & M2 to be (b,t,2h) which is currently (b,t,h)
        # FIXME: a should be (b,3,emb) which currently (b,3,h)
        # print(M2.shape)
        # print(q_encoder.shape)
        q_encoder = self.q_encoder_resize(q_encoder.transpose(1, 2)).transpose(1, 2)
        M2 = self.M2_resize(M2).transpose(1, 2)
        # Layer4: Prediction Layer
        loss = self.predictio_layer(q_encoder, M2, a_embedding, is_train=is_train, is_argmax=is_argmax)
        return loss"""

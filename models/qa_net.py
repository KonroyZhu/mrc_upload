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
        super(Highway, self).__init__()
        self.num = layer_num
        # 输入输出的隐层维度一致
        self.linear = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(self.num)])
        self.gate = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(self.num)])
        self.initiation()
    def initiation(self):
        for module in self.modules():
            if isinstance(module, nn.Linear): # 用0.1来限制，初始化所有nn.Linear的权重
                nn.init.xavier_uniform_(module.weight, 0.1)
    def forward(self, x):
        # x: (b,t,h) highway 中没有使用conv->无需将h放在中间作为in_channel
        for i in range(self.num):
            T = torch.sigmoid(self.gate[i](x))  # T as the transform gate (b,x,h) (h,h)
            H = F.relu(self.linear[i](x))  # H is an affine transform follow by activation
            C = (1 - T)  # (b,x,h)
            x = H * T + C * x  # (b,x,h)
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, conv_dim=1, bias=True): # 当k=5时可以用作projection
        super(DepthwiseSeparableConv, self).__init__()
        if conv_dim == 1:  # 一般先对输入转置 in_ch 为输入的hidden out_ch为输出的hidden
            # Input: (b, in_ch, x_i) | Output: (b,out_ch,x_o)
            # where x_o= [x_i + 2padding - (kernel_size-1) -1]/ stride +1
            self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch,
                                            kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch,
                                            kernel_size=1,  # 不使用groups
                                            padding=0, bias=bias)
        elif conv_dim == 2:
            # Input: (b, in_ch, x_i,y_i) | Output: (b,out_ch,x_o,y_o)
            # where x_o= [x_i + 2padding[0] - (kernel_size[0]-1) -1]/ stride[0] +1
            # and   y_o= [y_i + 2padding[1] - (kernel_size[1]-1) -1]/ stride[1] +1
            self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch,
                                            kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                                            kernel_size=1,  # 不使用groups
                                            padding=0, bias=bias)
        else:
            raise NotImplementedError
        # depthwise
        nn.init.kaiming_normal_(self.depthwise_conv.weight)  # 初始化depthwise conv权重
        nn.init.constant_(self.depthwise_conv.bias, 0.0)  # 0 初始化depthwise conv偏置
        # pointwise
        nn.init.kaiming_normal_(self.pointwise_conv.weight)
        nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        # x: (b,h,t)
        # (b,h,x) -> x + 0*2 - (1 - 1) + 5//2*2 - (5 - 1) -> x + 4 - 5 + 1 -> x
        x=self.depthwise_conv(x)
        x=self.pointwise_conv(x)
        return x  # 嵌套


class PosEncoder(nn.Module):
    def __init__(self, encoder_size):
        super(PosEncoder, self).__init__()
        self.hidden=encoder_size
        self.freqs = torch.Tensor([
            10000 ** -(i / encoder_size) if i % 2 == 0 \
                else 10000 ** -((i - 1) / encoder_size) \
            for i in range(encoder_size)
        ]).unsqueeze(1).cuda()  # (h,) -> (h,1)
        self.phases = torch.Tensor([
            0 if i % 2 == 0 \
                else math.pi / 2 \
            for i in range(encoder_size)
        ]).unsqueeze(1).cuda()  # (h,1) cos 与 sin 相差一个pi/2 phases 用于cos与sin的转换


    def forward(self, x,length):
        # x: (b,h,t)
        pos = torch.arange(length).repeat(self.hidden, 1).to(torch.float).cuda()  # (h,x)
        pos_encoding = nn.Parameter(
            torch.sin(
                torch.add(
                    torch.mul(pos, self.freqs), self.phases
                )
            ), requires_grad=False)  # (h,x)* (h,1) + (h,1) 处于1的维度自动广播-> (h,x)
        return x + pos_encoding  # (b,h,x)+(h,x) 自动广播->(b.h.x)


def mask_logits(target, mask):
    # mask为1的部分不变 为零的部分*（-1e30）
    return target * mask + (1 - mask) * (-1e30)


class SelfAttention(nn.Module):
    def __init__(self, encoder_size: int, num_header: int):
        super(SelfAttention, self).__init__()
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

    def forward(self, x, mask):
        # x: (b,h,t)
        # mask: (b,t,h)
        WQs, WKs, WVs = [], [], []
        dk = self.encoder_size / self.num_header
        sqrt_dk_inv = 1 / math.sqrt(dk)
        hmask = mask.unsqueeze(1)  # (b,1,t,h)
        vmask = mask.unsqueeze(2)  # (b,t,1,h)
        x = x.transpose(1, 2)  # (b,t,h)
        for i in range(self.num_header):
            WQs.append(torch.matmul(x, self.Wqs[1]))  # (b.t.h) (h,h/nh)
            WKs.append(torch.matmul(x, self.Wks[1]))
            WVs.append(torch.matmul(x, self.Wvs[1]))
        heads = []
        for i in range(self.num_header):
            out = torch.bmm(WQs[i], WKs[i].transpose(1, 2))  # 分母 公式(1) attention is all you need
            out = torch.mul(out, sqrt_dk_inv)  # *(1/分子) 公式(1) attention is all you need
            out = mask_logits(out, hmask)
            out = F.softmax(out, dim=2) * vmask
            headi = torch.bmm(out, WVs[i])  # 公式(1) softmax后乘V
            heads.append(headi)
        head = torch.cat(heads, dim=2)
        out = torch.matmul(head, self.Wo)
        return out.transpose(1, 2)

class Norm_Warpper(nn.Module):
    def __init__(self,in_ch,length):
        super(Norm_Warpper, self).__init__()
        self.norm = nn.LayerNorm([in_ch, length])
        self.length = length
    def forward(self, x):
        # x: (b,h,t)
        t = x.size(2)
        pad_length = int(self.length - t)  # 需要补充零的维度数
        out = F.pad(x, pad=(0, pad_length))  # 在倒数第一个维度补充pad_length列0 (b,h,length)
        out = self.norm(out)
        out = out[:,:,:t]
        return out


class Encoder_Block(nn.Module):
    def __init__(self, conv_num: int, in_ch: int, k: int, length: int, num_header=2, dropout=0.1):
        super(Encoder_Block, self).__init__()
        # in_ch 相当于输入的encoder_size
        self.length =length
        self.dropout = dropout
        self.num_conv = conv_num
        self.position_encoder = PosEncoder(encoder_size=in_ch)
        self.norm_begin = Norm_Warpper(in_ch, length)
        self.convs = nn.ModuleList([DepthwiseSeparableConv(in_ch=in_ch, out_ch=in_ch, k=k)
                                    for _ in range(conv_num)])
        self.norms = nn.ModuleList([Norm_Warpper(in_ch, length)
                                    for _ in range(conv_num)])
        self.self_att = SelfAttention(in_ch, num_header)
        self.norm_end = Norm_Warpper(in_ch, length)
        self.full_connect = nn.Linear(in_ch, in_ch, bias=True)

        self.initiation()

    def initiation(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):  # 用0.1来限制，初始化所有nn.Linear的权重
                nn.init.xavier_uniform_(module.weight, 0.1)
    def forward(self, x, mask):
        # x (b,h,x)
        # mask 用于区分id==0的<PAD>字符
        out = self.position_encoder(x,x.size(2))
        res = self.norm_begin(out)
        for i, conv in enumerate(self.convs):
            out = conv(out)  # 两层conv后输出形状一致
            out = F.relu(out)
            out = out + res
            if (i + 1) % 2 == 0:
                p_drop = self.dropout * (i + 1) / self.num_conv
                out = F.dropout(out, p_drop)
            res = out
            out = self.norms[i](out)
        out = self.self_att(out, mask)
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
        super(CQAttention, self).__init__()
        self.Wo = nn.Linear(3 * encoder_size, 1, bias=True)
        self.dropout = dropout
        nn.init.uniform_(self.Wo.weight, -math.sqrt(1 / encoder_size), math.sqrt(1 / encoder_size))

        self.initiation()

    def initiation(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):  # 用0.1来限制，初始化所有nn.Linear的权重
                nn.init.xavier_uniform_(module.weight, 0.1)
    def forward(self, C, Q, cmask, qmask):
        # C/cmask: (b,p,h)/(b,p)
        # Q/qmask: (b,q,h)/(b,q)
        ss = []
        cmask = cmask.unsqueeze(2)  # (b,p,1)
        qmask = qmask.unsqueeze(1)  # (b,1,q)

        Ct = C.unsqueeze(2)  # (b,p,1,h)
        Qt = Q.unsqueeze(1)  # (b,1,q,h)

        CQ = torch.mul(Ct, Qt)  # (b,p,q,h)
        shape = (C.size(0), C.size(1), Q.size(1), C.size(2))  # (b,p,q,h)
        S = torch.cat([Ct.repeat(1, 1, Q.size(1), 1), Qt.repeat(1, C.size(1), 1, 1), CQ], dim=3)  # (b,p,q,3h)
        S = self.Wo(S).view(S.size(0), S.size(1), -1)  # (b,p,q)
        Sq = F.softmax(mask_logits(S, qmask), dim=2)  # q attention
        Sc = F.softmax(mask_logits(S, cmask), dim=1)  # p attention
        A = torch.bmm(Sq, Q)  # (b,p,q) (b,q,h)-> (b,p,h)
        B = torch.bmm(torch.bmm(Sq, Sc.transpose(1, 2)), C)  # (b,p,q) (b,q,p) (b,p,h) -> (b,p,h)

        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)  # (b,p,4h)
        out = F.dropout(out, self.dropout)
        return out


class QA_Net(nn.Module):  # param:
    def initiation(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):  # 用0.1来限制，初始化所有nn.Linear的权重
                nn.init.xavier_uniform_(module.weight, 0.1)

    def __init__(self, options, embedding=None):
        super(QA_Net, self).__init__()
        self.opts = options
        # ## #print("loading fiedler")
        self.dep_info = pickle.load(open(options["dep_path"], "rb"))
        encoder_size = self.opts["hidden_size"]
        vocab_size = self.opts["vocab_size"]
        self.drop_out = options["dropout"]
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

        # k=1: o = i-+2*0-(k-1) = i -1 + 1 = i 用kernel=1来做projection保证除hidden_size外输出不变
        self.q_conv_project = nn.Conv1d(in_channels=embedding_size, out_channels=encoder_size, kernel_size=1)
        self.p_conv_project = nn.Conv1d(in_channels=embedding_size, out_channels=encoder_size, kernel_size=1)
        # self.a_conv_project = nn.Conv1d(in_channels=embedding_size, out_channels=encoder_size, kernel_size=1)
        self.q_highway = Highway(layer_num=2, hidden_size=encoder_size)
        self.p_highway = Highway(layer_num=2, hidden_size=encoder_size)
        self.a_highway = Highway(layer_num=2, hidden_size=embedding_size)

        self.q_conv_dws = DepthwiseSeparableConv(in_ch=encoder_size, out_ch=encoder_size, k=5)
        self.a_conv_dws = DepthwiseSeparableConv(in_ch=embedding_size, out_ch=embedding_size, k=5)
        self.p_conv_dws = DepthwiseSeparableConv(in_ch=encoder_size, out_ch=encoder_size, k=5)

        self.p_encoder = Encoder_Block(conv_num=4, in_ch=encoder_size, k=7, length=self.opts["p_len"])
        self.q_encoder = Encoder_Block(conv_num=4, in_ch=encoder_size, k=7, length=self.opts["q_len"])
        self.a_encoder = Encoder_Block(conv_num=4, in_ch=embedding_size, k=7, length=self.opts["alt_len"])

        # answer transform
        self.a_attention = nn.Linear(embedding_size, 1, bias=False)

        # Context-Query Attention
        self.cq_att = CQAttention(encoder_size)
        self.x_conv_project = DepthwiseSeparableConv(in_ch=4*encoder_size,out_ch=encoder_size,k=5)

        encoder_block = Encoder_Block(conv_num=2, in_ch=encoder_size, k=5, length=self.opts["p_len"])
        self.model_enc_blks = nn.ModuleList([encoder_block] * 7) # 7 个 encoder block

        # RESIZE
        self.q_encoder_resize = DepthwiseSeparableConv(in_ch=encoder_size, out_ch=2 * encoder_size, k=5)
        self.M2_resize = DepthwiseSeparableConv(in_ch=encoder_size, out_ch=2 *encoder_size, k=5)
        self.predictio_layer= Pred_Layer(options)

        self.initiation()

    def forward(self, inputs):
        [query, passage, answer, ids, is_train, is_argmax] = inputs
        opts = self.opts
        time_start = time.time()
        # Embedding & mask
        # TODO: mask 用于区分id是否为0 （id为0的是<PAD>）
        q_mask = (torch.zeros_like(query) != query).float()
        p_mask = (torch.zeros_like(passage) != passage).float()
        a_mask = (torch.zeros_like(answer) != answer).float().view(-1, answer.size(2))

        q_embedding = self.embedding(query)
        p_embedding = self.embedding(passage)
        a_embeddings = self.embedding(answer)
        a_embeddings = a_embeddings.view(-1, a_embeddings.size(2), a_embeddings.size(3))  # (3b,a,h)
        q_conv_projection = self.q_conv_project(q_embedding.transpose(1, 2)).transpose(1, 2)  # (b,q,emb)-> (b,q,h)
        p_conv_projection = self.p_conv_project(p_embedding.transpose(1, 2)).transpose(1, 2)  # (b,q,emb)-> (b,p,h)
        # a_conv_projection = self.a_conv_project(a_embeddings.transpose(1, 2)).transpose(1, 2)  # (b,q,emb)-> (3b,a,h)
        # #print("p/q_conv_projection: {}".format(p_conv_projection.shape))
        # #print("a_conv_projection: {}".format(a_conv_projection.shape))

        # # two-layer highway network
        q_highway = self.q_highway(q_conv_projection)
        p_highway = self.p_highway(p_conv_projection)
        a_highway = self.a_highway(a_embeddings)
        q_conv_dws = self.q_conv_dws(q_highway.transpose(1, 2)).transpose(1, 2)
        p_conv_dws = self.p_conv_dws(p_highway.transpose(1, 2)).transpose(1, 2)
        a_conv_dws = self.a_conv_dws(a_highway.transpose(1, 2)).transpose(1, 2)
        # #print("p/q_conv_dws: {}".format(p_conv_dws.shape))
        # #print("a_conv_dws: {}".format(a_conv_dws.shape))
        p_encoder = self.p_encoder(p_conv_dws.transpose(1, 2), p_mask).transpose(1, 2)
        q_encoder = self.q_encoder(q_conv_dws.transpose(1, 2), q_mask).transpose(1, 2)
        a_encoder = self.a_encoder(a_conv_dws.transpose(1, 2), a_mask).transpose(1, 2)
        # #print("p/q_encoder: {}".format(p_encoder.shape))
        # #print("a_encoder: {}".format(a_encoder.shape))
        # a score
        a_score = F.softmax(self.a_attention(a_encoder), 1)  # (3b,a,1)
        a_output = a_score.transpose(2, 1).bmm(a_encoder).squeeze()  # (3b,1,a) bmm (3b,a,h)-> (3b,1,h)
        a_embedding = a_output.view(answer.size(0), 3, -1)  # (b,3,h)
        time_start = time.time()
        X = self.cq_att(p_encoder, q_encoder, p_mask, q_mask)
        # #print("CQ(X): {}".format(X.shape)) # (b.q.4h)

        M1 = self.x_conv_project(X.transpose(1,2)) # (b,h,q)
        # #print("M1: {}".format(M1.shape))
        for enc in self.model_enc_blks: # 7个
            M1 = enc(M1, p_mask)
        M2 = M1
        # #print("M2: {}".format(M2))


        # FIXME: q_encoder & M2 to be (b,t,2h) which is currently (b,t,h)
        # FIXME: a should be (b,3,emb) which currently (b,3,h)
        # print(M2.shape)
        # print(q_encoder.shape)
        q_encoder=self.q_encoder_resize(q_encoder.transpose(1,2)).transpose(1,2)
        M2=self.M2_resize(M2).transpose(1,2)
        # Layer4: Prediction Layer
        loss = self.predictio_layer(q_encoder,M2,a_embedding,is_train=is_train,is_argmax=is_argmax)
        return loss



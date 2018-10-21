import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.pred_layer import Pred_Layer
from models.qa_utils import Highway, PosEncoder, LayerNorm_Buffer, SelfAttention, CQAttention


class Emb_RNN_Wrapper(nn.Module):
    def __init__(self, layer_num: int, emb_dim: int, hidden_size: int):
        super().__init__()
        self.rnn_projector = nn.GRU(input_size=emb_dim,hidden_size=hidden_size,bias=True,bidirectional=True)
        self.highway = Highway(layer_num, 2*hidden_size)

    def forward(self, x):
        # x: (b,t,h)
        _x,_ = self.rnn_projector(x)
        output = self.highway(_x)
        return output

class Encoder_RNN_Block(nn.Module):
    def __init__(self, fune_num: int, hidden_size: int, k: int, num_header=2, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.num_conv = fune_num
        self.position_encoder = PosEncoder(dropout=dropout, d_model=2*hidden_size)
        self.norm_begin = LayerNorm_Buffer()
        self.grus = nn.ModuleList(
            [nn.GRU(input_size=2*hidden_size,hidden_size=hidden_size,bidirectional=True,bias=True)
             for _ in range(fune_num)])  # t + 2*pad//2 - (k-1) (pad=k//2保证输出一致)
        self.norms = nn.ModuleList([LayerNorm_Buffer()
                                    for _ in range(fune_num)])
        self.self_att = SelfAttention(2*hidden_size, num_header)
        self.norm_end = LayerNorm_Buffer()
        self.full_connect = nn.Linear(2*hidden_size, 2*hidden_size, bias=True)

    def forward(self, x):
        # x (b,t,h)
        out = self.position_encoder(x)  # (b,t,h)
        res = self.norm_begin(out.transpose(1, 2)).transpose(1, 2) # (x,t,h)
        for i, gru in enumerate(self.grus):
            out,_ = gru(out)  # 两层conv后输出形状一致
            out = F.relu(out)
            # #print("out",out.shape)
            # #print("res",res.shape)
            out = out + res
            if (i + 1) % 2 == 0:
                p_drop = self.dropout * (i + 1) / self.num_conv
                # p_drop = self.dropout
                out = F.dropout(out, p_drop)
            res = out # (x,t,h)
            out = self.norms[i](out.transpose(1, 2)).transpose(1, 2) # (x,t,h)
        out = self.self_att(out.transpose(1, 2)).transpose(1, 2)
        out = out + res
        out = F.dropout(out, self.dropout)
        res = out
        out = self.norm_end(out.transpose(1, 2)).transpose(1, 2)
        out = self.full_connect(out)
        out = F.relu(out)
        out = out + res
        out = F.dropout(out, self.dropout)
        return out


class QA_RNN(nn.Module):
    def __init__(self,option,embedding=None):
        super(QA_RNN,self).__init__()
        self.opts = option
        self.drop_out = option["dropout"]
        vocab_size = self.opts["vocab_size"]
        encoder_size = self.opts["hidden_size"]
        if embedding is None:
            embedding_size = option["emb_size"]
            self.embedding = nn.Embedding(vocab_size + 1, embedding_dim=embedding_size)
            initrange = 0.1
            nn.init.uniform_(self.embedding.weight, -initrange, initrange)  # embedding初始化为-0.1~0.1之间
        else:
            embedding_size = np.shape(embedding)[1]  # (vocab_size,embedding_dim)
            self.embedding = nn.Embedding(vocab_size + 1, embedding_dim=embedding_size)
            self.embedding.from_pretrained(embedding, freeze=True)  # TODO:斟酌一下要不要freeze

        self.a_emb_wrap = Emb_RNN_Wrapper(layer_num=2, hidden_size=int(embedding_size/2), emb_dim=embedding_size)
        self.q_emb_wrap = Emb_RNN_Wrapper(layer_num=2, hidden_size=encoder_size, emb_dim=embedding_size)
        self.p_emb_wrap = Emb_RNN_Wrapper(layer_num=2, hidden_size=encoder_size, emb_dim=embedding_size)

        self.a_encoder = Encoder_RNN_Block(fune_num=4, hidden_size=int(embedding_size/2), k=7, dropout=self.drop_out)
        self.q_encoder = Encoder_RNN_Block(fune_num=4, hidden_size=encoder_size, k=7, dropout=self.drop_out)
        self.d_encoder = Encoder_RNN_Block(fune_num=4, hidden_size=encoder_size, k=7, dropout=self.drop_out)

        self.q_p_attention = CQAttention(2*encoder_size, self.drop_out)
        self.a_attention = nn.Linear(embedding_size, 1, bias=False)
        self.qp_projector = nn.GRU(input_size=8*encoder_size,hidden_size=encoder_size,bidirectional=True,bias=True)


        encoder_block = Encoder_RNN_Block(fune_num=2, hidden_size=encoder_size, k=5)
        self.model_enc_blks = nn.ModuleList([encoder_block] * 7)  # 7 个 encoder block

        self.prediction_layer = Pred_Layer(self.opts)
        self.initiation()

    def initiation(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):  # 用0.1来限制，初始化所有nn.Linear的权重
                print("initializing Linear:", module)
                nn.init.xavier_uniform_(module.weight, 0.1)

    def forward(self, inputs):
        [query, passage, answer, ids, is_train, is_argmax] = inputs
        # Embedding
        q_emb= self.embedding(query)
        p_emb= self.embedding(passage)
        a_emb = self.embedding(answer).view(answer.size(0) * 3, answer.size(2), q_emb.size(2))

        q_emb = self.q_emb_wrap(q_emb) # (b,t,2h)
        p_emb = self.p_emb_wrap(p_emb) # (b,t,2h)
        a_emb = self.a_emb_wrap(a_emb) # (b,t,2d)

        # Encoding a
        a_embedding = self.a_encoder(a_emb) # （3b,a,2d)
        a_score = F.softmax(self.a_attention(a_embedding), 1)  # (3b,a,1)
        a_output = a_score.transpose(2, 1).bmm(a_embedding).squeeze()  # (3b,1,a) bmm (3b,a,d)-> (3b,1,d)
        a_embedding = a_output.view(q_emb.size(0), 3, a_emb.size(2))  # (b,3,d)

        #   Encoder Block
        Q = self.q_encoder(q_emb)
        Q = F.dropout(Q, self.drop_out) # (b,p,2h)
        P = self.d_encoder(p_emb)
        P = F.dropout(P, self.drop_out)  # (b,p,2h)

        # QP Attention
        q_p_att = self.q_p_attention(P, Q)  # (b,p,8h)
        M1,_ = self.qp_projector(q_p_att)
        for enc in self.model_enc_blks:  # 7个
            M1 = enc(M1)
        M2 = M1
        # print("M2",M2)
        print(M2)
        print(Q)
        print(a_embedding)
        loss = self.prediction_layer(Q, M2, a_embedding, is_train=is_train, is_argmax=is_argmax)
        return loss

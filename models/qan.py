import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.pred_layer import Pred_Layer
from models.qa_utils import Encoder_Block, Emb_Wrapper, CQAttention


class QAN(nn.Module):  # param: 16821760
    def __init__(self, options, embedding=None):
        super(QAN, self).__init__()
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

        self.a_emb_wrap = Emb_Wrapper(layer_num=2, hidden_size=embedding_size, emb_dim=embedding_size)
        self.q_emb_wrap = Emb_Wrapper(layer_num=2, hidden_size=encoder_size, emb_dim=embedding_size)
        self.p_emb_wrap = Emb_Wrapper(layer_num=2, hidden_size=encoder_size, emb_dim=embedding_size)

        self.a_encoder = Encoder_Block(conv_num=4, in_channels=embedding_size, k=7, dropout=self.drop_out)
        self.q_encoder = Encoder_Block(conv_num=4, in_channels=encoder_size, k=7, dropout=self.drop_out)
        self.d_encoder = Encoder_Block(conv_num=4, in_channels=encoder_size, k=7, dropout=self.drop_out)

        self.q_conv_projector = nn.Conv1d(in_channels=encoder_size, out_channels=2 * encoder_size, padding=5 // 2,
                                          kernel_size=5)
        self.a_attention = nn.Linear(embedding_size, 1, bias=False)

        self.q_p_attention = CQAttention(encoder_size,self.drop_out)
        self.qp_projector = nn.Conv1d(in_channels=4 * encoder_size, out_channels=2 * encoder_size, padding=5 // 2,
                                      kernel_size=5)
        encoder_block = Encoder_Block(conv_num=2, in_channels=2*encoder_size, k=5)
        self.model_enc_blks = nn.ModuleList([encoder_block] * 7)  # 7 个 encoder block
        self.model_enc=nn.GRU(input_size=2*encoder_size,hidden_size=encoder_size,bidirectional=True,bias=True,dropout=self.drop_out)  #尝试替换encoderblock

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
        q_emb = self.embedding(query)
        p_emb = self.embedding(passage)
        a_emb = self.embedding(answer).view(answer.size(0) * 3, answer.size(2), q_emb.size(2))

        q_emb = self.q_emb_wrap(q_emb)
        p_emb = self.p_emb_wrap(p_emb)
        a_emb = self.a_emb_wrap(a_emb)

        # Encoding a
        a_embedding = self.a_encoder(a_emb.transpose(1, 2)).transpose(1, 2)  # （3b,a,h)
        a_score = F.softmax(self.a_attention(a_embedding), 1)  # (3b,a,1)
        a_output = a_score.transpose(2, 1).bmm(a_embedding).squeeze()  # (3b,1,a) bmm (3b,a,h)-> (3b,1,h)
        a_embedding = a_output.view(q_emb.size(0), 3, a_emb.size(2))  # (b,3,h)

        #   Encoder Block
        Q = self.q_encoder(q_emb.transpose(1, 2)).transpose(1,2)
        Q = F.dropout(Q, self.drop_out)
        P = self.d_encoder(p_emb.transpose(1, 2)).transpose(1,2)
        P = F.dropout(P, self.drop_out)  # (b,p,h)

        # QP Attention
        q_p_att=self.q_p_attention(P,Q) # (b,p,4h)
        # """       ccl
        M1 = self.qp_projector(q_p_att.transpose(1, 2)).transpose(1,2)
        M2,_ = self.model_enc(M1)     # 尝试替换encoder_block
        """            ccc
        for enc in self.model_enc_blks:  # 7个
            M1 = enc(M1)
        M2 = M1.transpose(1,2)
        # """
        Q = self.q_conv_projector(Q.transpose(1, 2)).transpose(1, 2) # 可以考虑一开始就project 到2h维
        loss = self.prediction_layer(Q, M2, a_embedding, is_train=is_train, is_argmax=is_argmax)
        return loss
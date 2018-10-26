import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.pred_layer import Pred_Layer
from models.qa_utils import Encoder_Block, Emb_Wrapper


class QA_DCN(nn.Module):  # param: 16821760
    def __init__(self, options, embedding=None):
        super(QA_DCN, self).__init__()
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

        self.a_emb_wrap = Emb_Wrapper(layer_num=2, hidden_size=encoder_size, emb_dim=embedding_size)
        self.q_emb_wrap = Emb_Wrapper(layer_num=2, hidden_size=encoder_size, emb_dim=embedding_size)
        self.p_emb_wrap = Emb_Wrapper(layer_num=2, hidden_size=encoder_size, emb_dim=embedding_size)

        self.a_encoder = Encoder_Block(conv_num=4, in_channels=encoder_size, k=7,dropout=self.drop_out)
        self.q_encoder = Encoder_Block(conv_num=4, in_channels=encoder_size, k=7,dropout=self.drop_out)
        self.d_encoder = Encoder_Block(conv_num=4, in_channels=encoder_size, k=7,dropout=self.drop_out)

        self.q_conv_projector = nn.Conv1d(in_channels=encoder_size, out_channels=2*encoder_size, padding=5 // 2, kernel_size=5)
        self.p_conv_projector = nn.Conv1d(in_channels=encoder_size, out_channels=2*encoder_size, padding=5 // 2, kernel_size=5)

        self.a_attention = nn.Linear(embedding_size, 1, bias=False)

        self.W_Q = nn.Linear(2 * encoder_size, 2 * encoder_size, bias=True)

        self.U_lstm = nn.LSTM(input_size=8 * encoder_size, hidden_size=encoder_size, batch_first=True,
                              bidirectional=True, bias=False)

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
        a_embedding = self.a_encoder(a_emb.transpose(1,2)).transpose(1,2)  # （3b,a,h)
        a_score = F.softmax(self.a_attention(a_embedding), 1)  # (3b,a,1)
        a_output = a_score.transpose(2, 1).bmm(a_embedding).squeeze()  # (3b,1,a) bmm (3b,a,h)-> (3b,1,h)
        a_embedding = a_output.view(q_emb.size(0), 3, a_emb.size(2))  # (b,3,h)

        #   1 DOCUMENT AND QUESTION ENCODER
        Q= self.q_encoder(q_emb.transpose(1,2))
        Q= self.q_conv_projector(Q).transpose(1,2)
        Q = F.dropout(Q,self.drop_out)
        D = self.d_encoder(p_emb.transpose(1,2))
        D = self.p_conv_projector(D).transpose(1,2)
        D = F.dropout(D, self.drop_out)  # (b,d,2h)

        #   2 COATTENTION ENCODER
        L = D.bmm(Q.transpose(2, 1))  # (b,d,h) bmm (b,h,q)
        # print("L: {}".format(np.shape(L)))  # (b,d,q)
        AQ = F.softmax(L, dim=2)  # (b,d,q)
        AD = F.softmax(L.transpose(2, 1), dim=2)  # (b,q,d)
        # print("AQ: {}".format(np.shape(AQ)))
        # print("AD: {}".format(np.shape(AD)))

        CQ = D.transpose(2, 1).bmm(AQ)  # (b,d,h) (b,d,q) -> (b,h,q)
        # print("CQ: {}".format(np.shape(CQ)))

        Q_CQ = torch.cat([Q, CQ.transpose(2, 1)], 2)  # (b.q.4h)
        # print("Q_CQ: {}".format(np.shape(Q_CQ)))

        CD = AD.transpose(2, 1).bmm(Q_CQ)  # (b,d,q) (b,q,4h) -> (b.d.4h)
        # print("CD: {}".format(np.shape(CD)))

        D_CD = torch.cat([torch.cat([D, D], 2), CD], 2)  # (b,d,4h)
        # print("D_CD: {}".format(np.shape(D_CD)))

        U, _ = self.U_lstm(D_CD)
        # print("U: {}".format(np.shape(U)))  # (b,d,2h)

        loss = self.prediction_layer(Q, U, a_embedding, is_train=is_train, is_argmax=is_argmax)
        return loss
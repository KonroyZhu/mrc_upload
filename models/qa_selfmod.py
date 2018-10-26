import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from com.train_utils import ans_shuffle
from models.pred_layer import Pred_Layer
from models.qa_utils import CQAttention, PosEncoder, SelfAttention


class QA_SelfM(nn.Module):  # param: 16821760
    def __init__(self, options, embedding=None):
        super(QA_SelfM, self).__init__()
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

        self.a_encoder = nn.LSTM(input_size=embedding_size, hidden_size=int(embedding_size / 2), batch_first=True,
                                 bias=False, bidirectional=True, dropout=self.drop_out)
        self.q_encoder = nn.LSTM(input_size=embedding_size, hidden_size=encoder_size, batch_first=True,
                                 bidirectional=True,
                                 bias=False, dropout=self.drop_out)
        self.d_encoder = nn.LSTM(input_size=embedding_size, hidden_size=encoder_size, batch_first=True,
                                 bidirectional=True,
                                 bias=False, dropout=self.drop_out)

        self.a_attention = nn.Linear(embedding_size, 1, bias=False)

        self.W_Q = nn.Linear(2 * encoder_size, 2 * encoder_size, bias=True)
        self.q_p_attention = CQAttention(2 * encoder_size, self.drop_out)


        self.U_lstm = nn.LSTM(input_size=8 * encoder_size, hidden_size=encoder_size, batch_first=True,
                              bidirectional=True, bias=False)
        self.self_att = SelfAttention(encoder_size=2 * encoder_size, num_header=4)
        """
        prediction layer
        """

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
        d_emb = self.embedding(passage)
        a_emb = self.embedding(answer)

        # Layer1: Encoding Layer
        # Encoding a
        a_embedding, _ = self.a_encoder(a_emb.view(-1, a_emb.size(2), a_emb.size(3)))  # （3b,a,h)
        a_score = F.softmax(self.a_attention(a_embedding), 1)  # (3b,a,1)
        a_output = a_score.transpose(2, 1).bmm(a_embedding).squeeze()  # (3b,1,a) bmm (3b,a,h)-> (3b,1,h)
        a_embedding = a_output.view(a_emb.size(0), 3, -1)  # (b,3,h)

        #   DYNAMIC COATTENTION NETWORKS
        #   1 DOCUMENT AND QUESTION ENCODER
        Q_, _ = self.q_encoder(q_emb)
        Q_ = F.dropout(Q_, self.drop_out)  # (b,q,2h)
        Q = F.tanh(self.W_Q(Q_))
        D, _ = self.d_encoder(d_emb)
        D = F.dropout(D, self.drop_out)  # (b,d,2h)

        #   cq attention
        q_p_att = self.q_p_attention(D, Q)
        # self_att = self.self_att(q_p_att.transpose(1, 2)).transpose(1, 2) # FIXME: fake self model
        U, _ = self.U_lstm(q_p_att) # (b,d,2h)
        self_att = self.self_att(U)
        # print("U: {}".format(np.shape(U)))  # (b,d,2h)

        loss = self.prediction_layer(Q, self_att, a_embedding, is_train=is_train, is_argmax=is_argmax,print_score=True)
        return loss

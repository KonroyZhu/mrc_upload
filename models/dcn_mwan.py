import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.pred_layer import Pred_Layer


class DM(nn.Module):  # param: 16821760
    def __init__(self, options, embedding=None):
        super(DM, self).__init__()
        self.drop_out = options["dropout"]
        self.opts=options
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
                                 bias=False,bidirectional=True)
        self.q_encoder = nn.LSTM(input_size=embedding_size, hidden_size=encoder_size, batch_first=True,bidirectional=True,
                                 bias=False)
        self.d_encoder = nn.LSTM(input_size=embedding_size, hidden_size=encoder_size, batch_first=True,bidirectional=True,
                                       bias=False)

        self.a_attention = nn.Linear(embedding_size, 1, bias=False)

        self.W_Q = nn.Linear(2*encoder_size, 2*encoder_size, bias=True)

        self.U_lstm = nn.LSTM(input_size=8*encoder_size,hidden_size=encoder_size,batch_first=True,bidirectional=True,bias=False)

        """
        prediction layer
        """
        self.Wq = nn.Linear(encoder_size, encoder_size, bias=False)
        self.vq = nn.Linear(encoder_size, 1, bias=False)
        self.Wp1 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wp2 = nn.Linear( encoder_size, encoder_size, bias=False)
        self.vp = nn.Linear(encoder_size, 1, bias=False)
        self.prediction = nn.Linear(2 * encoder_size, embedding_size, bias=False)

        """
                multi-way attention layer
                """
        # Concat Attention
        self.Wc1 = nn.Linear(2 * self.opts["hidden_size"], self.opts["hidden_size"], bias=False)
        self.Wc2 = nn.Linear(2 * self.opts["hidden_size"], self.opts["hidden_size"], bias=False)
        self.vc = nn.Linear(self.opts["hidden_size"], 1, bias=False)
        # Bilinear Attention
        self.Wb = nn.Linear(2 * self.opts["hidden_size"], 2 * self.opts["hidden_size"], bias=False)
        # Dot Attention :
        self.Wd = nn.Linear(2 * self.opts["hidden_size"], self.opts["hidden_size"], bias=False)
        self.vd = nn.Linear(self.opts["hidden_size"], 1, bias=False)
        # Minus Attention :
        self.Wm = nn.Linear(2 * self.opts["hidden_size"], self.opts["hidden_size"], bias=False)
        self.vm = nn.Linear(self.opts["hidden_size"], 1, bias=False)

        self.Ws = nn.Linear(2 * self.opts["hidden_size"], self.opts["hidden_size"], bias=False)
        self.vs = nn.Linear(self.opts["hidden_size"], 1, bias=False)

        self.gru_agg = nn.GRU(2 * self.opts["hidden_size"], self.opts["hidden_size"], batch_first=True,
                              bidirectional=True)
        self.out_gru = nn.GRU(4*encoder_size,encoder_size,bidirectional=True,batch_first=True,dropout=self.drop_out)
        """
       aggregating layer
       """
        self.Wgc = nn.Linear(4 * self.opts["hidden_size"], 4 * self.opts["hidden_size"], bias=False)
        self.gru_htc = nn.GRU(input_size=4 * self.opts["hidden_size"], hidden_size=self.opts["hidden_size"],
                              batch_first=True, bidirectional=True)

        self.Wgb = nn.Linear(4 * self.opts["hidden_size"], 4 * self.opts["hidden_size"], bias=False)
        self.gru_htb = nn.GRU(input_size=4 * self.opts["hidden_size"], hidden_size=self.opts["hidden_size"],
                              batch_first=True, bidirectional=True)

        self.Wgd = nn.Linear(4 * self.opts["hidden_size"], 4 * self.opts["hidden_size"], bias=False)
        self.gru_htd = nn.GRU(input_size=4 * self.opts["hidden_size"], hidden_size=self.opts["hidden_size"],
                              batch_first=True, bidirectional=True)

        self.Wgm = nn.Linear(4 * self.opts["hidden_size"], 4 * self.opts["hidden_size"], bias=False)
        self.gru_htm = nn.GRU(input_size=4 * self.opts["hidden_size"], hidden_size=self.opts["hidden_size"],
                              batch_first=True, bidirectional=True)

        self.W_agg = nn.Linear(2 * self.opts["hidden_size"], self.opts["hidden_size"], bias=False)
        self.v_agg = nn.Linear(self.opts["hidden_size"], 1, bias=False)

        self.prediction_layer=Pred_Layer(self.opts)
        self.initiation()

    def initiation(self):
        for module in self.modules():
            if isinstance(module, nn.Linear): # 用0.1来限制，初始化所有nn.Linear的权重
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
        # DCN###################################################################################
        #   2 COATTENTION ENCODER
        L = D.bmm(Q.transpose(2, 1))  # (b,d,h) bmm (b,h,q)
        # print("L: {}".format(np.shape(L)))  # (b,d,q)
        AQ=F.softmax(L,dim=2) # (b,d,q)
        AD=F.softmax(L.transpose(2,1),dim=2) # (b,q,d)
        # print("AQ: {}".format(np.shape(AQ)))
        # print("AD: {}".format(np.shape(AD)))

        CQ=D.transpose(2,1).bmm(AQ) # (b,d,h) (b,d,q) -> (b,h,q)
        # print("CQ: {}".format(np.shape(CQ)))

        Q_CQ=torch.cat([Q,CQ.transpose(2,1)],2) # (b.q.4h)
        # print("Q_CQ: {}".format(np.shape(Q_CQ)))

        CD=AD.transpose(2,1).bmm(Q_CQ)  # (b,d,q) (b,q,4h) -> (b.d.4h)
        # print("CD: {}".format(np.shape(CD)))

        D_CD=torch.cat([torch.cat([D,D],2),CD],2) # (b,d,4h)
        # print("D_CD: {}".format(np.shape(D_CD)))
        # MwAN###################################################################################
        # Layer2: Multi-attention Layer
        # (1) concatenate attention
        _s1 = self.Wc1(Q).unsqueeze(1)
        _s2 = self.Wc2(D).unsqueeze(2)
        sjt = self.vc(torch.tanh(_s1 + _s2)).squeeze()
        ait = F.softmax(sjt, 2)
        qtc = ait.bmm(Q)
        # (2) bi-linear attention
        _s1 = self.Wb(Q).transpose(2, 1)
        sjt = D.bmm(_s1)
        ait = F.softmax(sjt, 2)
        qtb = ait.bmm(Q)
        # (3) dot attention
        _s1 = Q.unsqueeze(1)
        _s2 = D.unsqueeze(2)
        sjt = self.vd(torch.tanh(self.Wd(_s1 * _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        qtd = ait.bmm(Q)
        # (4) minus attention
        sjt = self.vm(torch.tanh(self.Wm(_s1 - _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        qtm = ait.bmm(Q)
        # (5) self attention
        _s1 = D.unsqueeze(1)
        _s2 = D.unsqueeze(2)
        sjt = self.vs(torch.tanh(self.Ws(_s1 * _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        qts = ait.bmm(D)

        # Layer3: Aggregating Layer
        # (1) concatenate
        xtc = torch.cat([qtc, D], 2)  # (b,p,4h)
        gtc = torch.sigmoid(self.Wgc(xtc))  # (b,p,4h)
        xtc_star = gtc * xtc  # (b,p,4h)
        htc, _ = self.gru_htc(xtc_star)  # (b,p,2h) due to bi-directional
        # (2) bi-linear
        xtb = torch.cat([qtb, D], 2)
        gtb = torch.sigmoid(self.Wgb(xtb))
        xtb_star = gtb * xtb
        htb, _ = self.gru_htb(xtb_star)  # (b,p,2h)
        # (3) dot
        xtd = torch.cat([qtd, D], 2)
        gtd = torch.sigmoid(self.Wgd(xtd))
        xtd_star = gtd * xtd
        htd, _ = self.gru_htd(xtd_star)  # (b,p,2h)
        # (4) minus
        xtm = torch.cat([qtm, D], 2)
        gtm = torch.sigmoid(self.Wgm(xtm))
        xtm_star = gtm * xtm
        htm, _ = self.gru_htm(xtm_star)  # (b,p,2h)

        # attention
        aggregation = torch.cat([htc, htb, htd, htm], 2)  # (b,p,8h)
        aggregation = aggregation.view(aggregation.size(0), aggregation.size(1), 4, -1)  # (b,p,4,2h)
        sjt = self.v_agg(torch.tanh(self.W_agg(aggregation)))  # (b,p,4,1)
        sjt = sjt.view(sjt.size(0) * sjt.size(1), 4, -1).transpose(2, 1)  # (bp,4,1) -> (bp,1,4)
        ait = F.softmax(sjt, dim=2)
        rep = ait.bmm(
            aggregation.view(aggregation.size(0) * aggregation.size(1), 4, -1))  # (bp,1,4) bmm (bp,4,2h) ->(bp,1,2h)
        rep = rep.view(aggregation.size(0), aggregation.size(1), 1, -1).squeeze()  # (b,p,2h)

        # Final###################################################################################
        U,_ = self.U_lstm(D_CD)
        aggregation_representation, _ = self.gru_agg(rep)  # (b,p,2h)
        output = torch.cat([U,aggregation_representation],2) # (b,p,4h)
        output,_ = self.out_gru(output)
        # print("U: {}".format(np.shape(U)))  # (b,d,2h)

        loss = self.prediction_layer(Q,U,a_embedding,is_train=is_train,is_argmax=is_argmax)
        return loss
        # 3: Prediction Layer
        # Layer4: Prediction Layer
        # sj = self.vq(torch.tanh(self.Wq(Q))).transpose(2, 1)  # (b,q,h) (h,h) (h,1) -> (b,q,1) -> (b,1,q)
        # rq = F.softmax(sj, 2).bmm(Q)  # (b,1,q) (b,q,h) -> (b,1,h)
        # sj = F.softmax(self.vp(self.Wp1(U) + self.Wp2(rq)).transpose(2, 1), 2)
        # rp = sj.bmm(U)
        #
        # encoder_output = F.dropout(F.leaky_relu(self.prediction(rp)), self.drop_out)
        # score = F.softmax(a_embedding.bmm(encoder_output.transpose(2, 1)).squeeze(), 1)
        # if not is_train:
        #     return score.argmax(1)
        # loss = -torch.log(score[:, 0]).mean()
        # return loss

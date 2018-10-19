from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.pred_layer import Pred_Layer


class Mw_f_ori(nn.Module): # param: 16821760
    def __init__(self, options,embedding): # FIXME
        super(Mw_f_ori, self).__init__()
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

        self.q_encoder = nn.GRU(input_size=embedding_size, hidden_size=encoder_size, batch_first=True,
                                bidirectional=True)
        self.p_encoder = nn.GRU(input_size=embedding_size, hidden_size=encoder_size, batch_first=True,
                                bidirectional=True)
        self.a_encoder = nn.GRU(input_size=embedding_size, hidden_size=int(embedding_size / 2), batch_first=True,
                                bidirectional=True) # GRU的input_size与hidden_size要为相同类型，此处除2后变成了非int
        self.a_attention = nn.Linear(embedding_size, 1, bias=False)

        """
        multi-way attention layer
        """
        # Concat Attention
        self.Wc1 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wc2 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vc = nn.Linear(encoder_size, 1, bias=False)
        # Bilinear Attention
        self.Wb = nn.Linear(2 * encoder_size, 2 * encoder_size, bias=False)
        # Dot Attention :
        self.Wd = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vd = nn.Linear(encoder_size, 1, bias=False)
        # Minus Attention :
        self.Wm = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vm = nn.Linear(encoder_size, 1, bias=False)

        self.Ws = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vs = nn.Linear(encoder_size, 1, bias=False)

        self.gru_agg = nn.GRU(2*encoder_size, encoder_size, batch_first=True, bidirectional=True)

        """
       aggregating layer
       """
        self.Wgc=nn.Linear(4*encoder_size,4*encoder_size,bias=False)
        self.gru_htc=nn.GRU(input_size=4*encoder_size,hidden_size=encoder_size,batch_first=True,bidirectional=True)

        self.Wgb = nn.Linear(4 * encoder_size, 4 * encoder_size, bias=False)
        self.gru_htb = nn.GRU(input_size=4 * encoder_size, hidden_size=encoder_size, batch_first=True, bidirectional=True)

        self.Wgd = nn.Linear(4 * encoder_size, 4 * encoder_size, bias=False)
        self.gru_htd = nn.GRU(input_size=4 * encoder_size, hidden_size=encoder_size, batch_first=True, bidirectional=True)

        self.Wgm = nn.Linear(4 * encoder_size, 4 * encoder_size, bias=False)
        self.gru_htm = nn.GRU(input_size=4 * encoder_size, hidden_size=encoder_size, batch_first=True, bidirectional=True)

        self.W_agg=nn.Linear(2*encoder_size,encoder_size,bias=False)
        self.v_agg=nn.Linear(encoder_size,1,bias=False)

        """
        prediction layer
        """
        self.Wq = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vq = nn.Linear(encoder_size, 1, bias=False)
        self.Wp1 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wp2 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vp = nn.Linear(encoder_size, 1, bias=False)
        self.prediction = nn.Linear(2 * encoder_size, embedding_size, bias=False)
        self.initiation()

    def initiation(self):
        for module in self.modules():
            if isinstance(module, nn.Linear): # 用0.1来限制，初始化所有nn.Linear的权重
                print("initializing Linear:", module)
                nn.init.xavier_uniform_(module.weight, 0.1)

    def forward(self, inputs):
        [query, passage, answer, ids, is_train, is_argmax] = inputs
        # Embedding
        q_embedding = self.embedding(query)
        p_embedding = self.embedding(passage)
        a_embeddings = self.embedding(answer)
        # Layer1: Encoding Layer
        # Encoding a
        a_embedding, _ = self.a_encoder(a_embeddings.view(-1, a_embeddings.size(2), a_embeddings.size(3)))  # （3b,a,h)
        a_score = F.softmax(self.a_attention(a_embedding), 1) # (3b,a,1)
        a_output = a_score.transpose(2, 1).bmm(a_embedding).squeeze() # (3b,1,a) bmm (3b,a,h)-> (3b,1,h)
        a_embedding = a_output.view(a_embeddings.size(0), 3, -1)  # (b,3,h)
        # Encoding p,q
        hq, _ = self.q_encoder(p_embedding)
        hq=F.dropout(hq,self.drop_out)  # (b,q,2h)
        hp, _ = self.p_encoder(q_embedding)
        hp=F.dropout(hp,self.drop_out)  # (b,p,2h)

        # Layer2: Multi-attention Layer
        # (1) concatenate attention
        _s1 = self.Wc1(hq).unsqueeze(1)
        _s2 = self.Wc2(hp).unsqueeze(2)
        sjt = self.vc(torch.tanh(_s1 + _s2)).squeeze()
        ait = F.softmax(sjt, 2)
        qtc = ait.bmm(hq)
        # (2) bi-linear attention
        _s1 = self.Wb(hq).transpose(2, 1)
        sjt = hp.bmm(_s1)
        ait = F.softmax(sjt, 2)
        qtb = ait.bmm(hq)
        # (3) dot attention
        _s1 = hq.unsqueeze(1)
        _s2 = hp.unsqueeze(2)
        sjt = self.vd(torch.tanh(self.Wd(_s1 * _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        qtd = ait.bmm(hq)
        # (4) minus attention
        sjt = self.vm(torch.tanh(self.Wm(_s1 - _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        qtm = ait.bmm(hq)
        # (5) self attention
        _s1 = hp.unsqueeze(1)
        _s2 = hp.unsqueeze(2)
        sjt = self.vs(torch.tanh(self.Ws(_s1 * _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        qts = ait.bmm(hp)

        # Layer3: Aggregating Layer
        # (1) concatenate
        xtc=torch.cat([qtc,hp],2) # (b,p,4h)
        gtc=torch.sigmoid(self.Wgc(xtc)) # (b,p,4h)
        xtc_star=gtc*xtc # (b,p,4h)
        htc,_=self.gru_htc(xtc_star)  # (b,p,2h) due to bi-directional
        # (2) bi-linear
        xtb=torch.cat([qtb,hp],2)
        gtb=torch.sigmoid(self.Wgb(xtb))
        xtb_star=gtb*xtb
        htb,_=self.gru_htb(xtb_star)  # (b,p,2h)
        # (3) dot
        xtd=torch.cat([qtd,hp],2)
        gtd=torch.sigmoid(self.Wgd(xtd))
        xtd_star=gtd*xtd
        htd,_=self.gru_htd(xtd_star)  # (b,p,2h)
        # (4) minus
        xtm=torch.cat([qtm,hp],2)
        gtm=torch.sigmoid(self.Wgm(xtm))
        xtm_star=gtm*xtm
        htm,_=self.gru_htm(xtm_star)  # (b,p,2h)

        # attention
        aggregation = torch.cat([htc,htb,htd,htm], 2)  # (b,p,8h)
        aggregation = aggregation.view(aggregation.size(0),aggregation.size(1),4,-1) # (b,p,4,2h)
        sjt=self.v_agg(torch.tanh(self.W_agg(aggregation))) # (b,p,4,1)
        sjt=sjt.view(sjt.size(0)*sjt.size(1),4,-1).transpose(2,1) # (bp,4,1) -> (bp,1,4)
        ait=F.softmax(sjt,dim=2)
        rep=ait.bmm(aggregation.view(aggregation.size(0)*aggregation.size(1),4,-1)) # (bp,1,4) bmm (bp,4,2h) ->(bp,1,2h)
        rep=rep.view(aggregation.size(0),aggregation.size(1),1,-1).squeeze() # (b,p,2h)
        aggregation_representation, _ = self.gru_agg(rep)  # (b,p,2h)

        # loss = self.pred_layer( hq,aggregation_representation,a_embedding,is_train=is_train,is_argmax=True)
        # return loss
        # Layer4: Prediction Layer
        sj = self.vq(torch.tanh(self.Wq(hq))).transpose(2, 1)   # (b,q,2h) (2h,h) (h,1) -> (b,q,1) -> (b,1,q)
        rq = F.softmax(sj, 2).bmm(hq) # (b,1,q) (b,q,2h) -> (b,1,2h)
        #  利用rq与agg训练attention权重sj
        sj = F.softmax(self.vp(self.Wp1(aggregation_representation) + self.Wp2(rq)).transpose(2, 1), 2) # (b,1,p)
        rp = sj.bmm(aggregation_representation)  # (b,1,p) (b,p,2h) -> (b,1,2h)
        # MLP
        encoder_output = F.dropout(F.leaky_relu(self.prediction(rp)),self.drop_out)  # (b,1,d)
        score = F.softmax(a_embedding.bmm(encoder_output.transpose(2, 1)).squeeze(), 1)  # (b,3,h) (b,h,1) -> (b,3)

        _score = np.around(score.cpu().detach().numpy(), decimals=2)
        print("score sample: {} {}".format(_score[0], _score[1]))
        print("batch score: {}".format(Counter(score.argmax(1).cpu().data.numpy())[0] / self.opts["batch"]))
        if not is_train:
            if is_argmax:
                return score.argmax(1)
            else:
                return score
        loss = -torch.log(score[:, 0]).mean()
        return loss


from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Pred_Layer(nn.Module):
    def __init__(self,opts,max_margin=False):
        super().__init__()
        self.opts=opts
        self.max_margin = max_margin
        self.Wq = nn.Linear(2 * opts["hidden_size"], opts["hidden_size"], bias=False)
        self.vq = nn.Linear(opts["hidden_size"], 1, bias=False)
        self.Wp1 = nn.Linear(2 * opts["hidden_size"], opts["hidden_size"], bias=False)
        self.Wp2 = nn.Linear(2 * opts["hidden_size"], opts["hidden_size"], bias=False)
        self.vp = nn.Linear(opts["hidden_size"], 1, bias=False)
        self.prediction = nn.Linear(2 * opts["hidden_size"], opts["emb_size"], bias=False)


    def forward(self, q_encoder,aggregation,a_embedding,is_train=True,is_argmax=True,print_score=False,labels=None):
        # q_encoder: (b,q,2h)
        # aggregation: (b,p,2h)
        # a_embedding: (b,3,h)
        sj = self.vq(torch.tanh(self.Wq(q_encoder))).transpose(2, 1)  # (b,q,2h) (2h,h) (h,1) -> (b,q,1) -> (b,1,q)
        rq = F.softmax(sj, 2).bmm(q_encoder)  # (b,1,q) (b,q,2h) -> (b,1,2h)
        #  利用rq与agg训练attention权重sj
        sj = F.softmax(self.vp(self.Wp1(aggregation) + self.Wp2(rq)).transpose(2, 1), 2)  # (b,1,p)
        rp = sj.bmm(aggregation)  # (b,1,p) (b,p,2h) -> (b,1,2h)
        # MLP
        encoder_output = F.dropout(F.leaky_relu(self.prediction(rp)), self.opts["dropout"])  # (b,1,d)
        score = F.softmax(a_embedding.bmm(encoder_output.transpose(2, 1)).squeeze(), 1)  # (b,3,h) (b,h,1) -> (b,3)
        if "nan" in str(score):
            print("nan !!")
        if print_score:
            _score = np.around(score.cpu().detach().numpy(), decimals=2)
            print("score sample: {} {}".format(_score[0], _score[1]))
            print("batch score: {}".format(Counter(score.argmax(1).cpu().data.numpy())[0] / self.opts["batch"]))
        if not is_train:
            if is_argmax:
                return score.argmax(1)
            else:
                return score
        if labels is None:
            correct_answer = score[:, 0]
        else:
            assert  len(labels) == score.shape[0] #FIXME: 取score中的lables索引,labels 长度为batcch
            placeholder = np.arange(0,len(labels))
            correct_answer = score[placeholder,labels]

        if self.max_margin:
            """we take the maximum over i so that  we  are  ranking  the  correct  answer  over  the  best-ranked
                    incorrect answer (of which there are three) """
            correct = correct_answer
            m_score = torch.max(score, dim=1)[0]
            # #print(m_score.shape)
            u = 1.5  # score0 与 错误得分的间距
            margin = u + correct - m_score
            # #print(margin)
            zeros = torch.FloatTensor(np.zeros(shape=m_score.size(0)))  # fixme

            L = torch.max(zeros, margin.cpu())
            loss = L.mean()  # 最大化score0与错误选项的间距
        else:
            loss = -torch.log(correct_answer).mean()  # 原loss最大化score[0]的得分
        return loss
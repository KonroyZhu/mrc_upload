import pickle
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from com.utils import padding


class PHN(nn.Module):  # param: 16821760
    def __init__(self, options, embedding=None):
        super(PHN, self).__init__()
        self.opts = options
        # #print("loading fiedler")
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

        self.q_encoder = nn.GRU(input_size=embedding_size, hidden_size=encoder_size, batch_first=True,
                                bidirectional=True)
        self.p_encoder = nn.GRU(input_size=embedding_size, hidden_size=encoder_size, batch_first=True,
                                bidirectional=True)
        self.a_encoder = nn.GRU(input_size=embedding_size, hidden_size=encoder_size, batch_first=True,
                                bidirectional=True)
        self.a_attention = nn.Linear(2 * encoder_size, 1, bias=False)

        # 4.1 Semantic Perspectiv
        self.V_t = nn.Linear(2 * encoder_size, 1, bias=False)
        self.W_A_t = nn.Linear(2 * encoder_size, encoder_size, bias=True)

        self.V_h = nn.Linear(2 * encoder_size, 1, bias=False)
        self.W_A_h = nn.Linear(2 * encoder_size, encoder_size, bias=True)

        # 4.2 Word-by-Word Perspective
        self.W_B_t = nn.Linear(2 * encoder_size, encoder_size, bias=True)
        self.W_B_q = nn.Linear(2 * encoder_size, encoder_size, bias=True)
        self.W_B_a = nn.Linear(2 * encoder_size, encoder_size, bias=True)

        self.V_q = nn.Linear(2 * encoder_size, 1, bias=False)
        self.V_a = nn.Linear(2 * encoder_size, 1, bias=False)

        # 4.2.1 Sentential
        self.W_a1 = nn.Linear(3, 3, bias=False)
        self.W_a2 = nn.Linear(3, 3, bias=False)
        self.W_a3 = nn.Linear(3, 3, bias=False)

        # 4.2.2  Sequential Sliding Window
        # Gaussian distribution
        self.position_t = torch.rand(options["p_len"])

        self.W_a4 = nn.Linear(3, 3, bias=False)
        self.W_a5 = nn.Linear(3, 3, bias=False)
        self.W_a6 = nn.Linear(3, 3, bias=False)

        # 4.2.3 Dependency Sliding Window
        # Dependency sorting on text
        self.position_t2 = torch.rand(options["p_len"])

        self.W_a7 = nn.Linear(3, 3, bias=False)
        self.W_a8 = nn.Linear(3, 3, bias=False)
        self.W_a9 = nn.Linear(3, 3, bias=False)

        # 4.4 Combining Perspectives
        # self.W_mlp = nn.Linear(12, 3)
        self.W_mlp = nn.Linear(12, 2 * encoder_size)

        # prediction layer
        self.MLP = nn.Linear(12, 2 * encoder_size, bias=False)

        self.initiation()

    def initiation(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):  # 用0.1来限制，初始化所有nn.Linear的权重
                nn.init.xavier_uniform_(module.weight, 0.1)

    def forward(self, inputs):
        try:
            [query, passage, answer, ids, is_train, is_argmax] = inputs
            opts = self.opts
            # Embedding
            q_embedding = self.embedding(query)
            p_embedding = self.embedding(passage)
            a_embeddings = self.embedding(answer)
            # Layer1: Encoding Layer
            a, _ = self.a_encoder(a_embeddings.view(-1, a_embeddings.size(2), a_embeddings.size(3)))
            a = F.dropout(a, self.drop_out)  # (b,a,2h)
            q, _ = self.q_encoder(q_embedding)
            q = F.dropout(q, self.drop_out)  # (b,q,2h)
            t, _ = self.p_encoder(p_embedding)
            t = F.dropout(t, self.drop_out)  # (b,p,2h)

            a_score = F.softmax(self.a_attention(a), 1)  # (3b,a,1)
            # # #print(a_score.shape)
            a_output = a_score.transpose(2, 1).bmm(a).squeeze()  # (3b,1,a) bmm (3b,a,2h)-> (3b,2h)
            # # #print(a_output.shape)
            a_emb = a_output.view(opts["batch"], 3, a.size(2))  # (b,3,2h)

            # 4.1 Semantic Perspective
            # text
            w_k_t = self.V_t(t)
            # #print("w_k_t: {}".format(np.shape(w_k_t)))  # (b,p,1)
            t_sum = w_k_t.transpose(2, 1).bmm(t).squeeze()  # (b,1,p) (b,p,2h) -> (b,2h)
            # #print("t_sum:{}".format(np.shape(t_sum)))
            st = F.leaky_relu(self.W_A_t(t_sum))
            # #print("st: {} ".format(np.shape(st)))  # (b,h)

            # hypothesis
            a = a.view(-1, 3, a.size(1), a.size(2))
            # #print("a: {}".format(np.shape(a)))

            q = q.unsqueeze(1).repeat(1, 3, 1, 1)  # (b,q,2h)->(b,1,q,2h)->(b,3,q,2h) 采用向量方式代替循环:将q和t在1维度上重复
            t = t.unsqueeze(1).repeat(1, 3, 1, 1)  # (b,3,t,2h)
            # #print("q/t: {}".format(np.shape(q)))
            h = torch.cat([q, a], dim=2)
            # #print("h: {}".format(np.shape(h)))  # (b,3,q+a,2h)
            w_k_h = self.V_h(h)
            # #print("w_k_h: {}".format(np.shape(w_k_h)))  # (b,3,q+a,1)
            h_sum = w_k_h.view(h.size(0) * 3, h.size(2), -1).transpose(2, 1).bmm(
                h.view(h.size(0) * 3, h.size(2), -1)).squeeze()
            h_sum = h_sum.view(h.size(0), 3, h.size(3))
            # #print("h_sum: {}".format(np.shape(h_sum)))  # (3,b,1,q+a) (3,b,q+a,2h) -> (3,b,2h)
            sh = F.leaky_relu(self.W_A_h(h_sum))  # (3,b,2h) (2h,h) -> (3,b,h)
            # #print("sh: {}".format(np.shape(sh)))  # (b,3,h)

            st = st.unsqueeze(1).repeat(1, 3, 1)
            # #print("st: {}".format(np.shape(st)))  # (3,b,h)
            # #print("sh: {}".format(np.shape(sh)))  # (3,b,h)
            M_sem = F.cosine_similarity(st, sh, dim=2)
            M_sem = F.dropout(M_sem, self.drop_out)

            # #print("--Semantic-- M_sem: {}".format(np.shape(M_sem)))  # (3,b)

            # 4.2 Word-by-Word Perspective
            def get_position(pos_weight):
                """

                :param pos_weight: (t,)
                :return:
                """
                position_T = pos_weight[:t.size(2)]
                position_T = position_T.unsqueeze(0).unsqueeze(1).unsqueeze(3).repeat(opts["batch"], 3, 1, 1)
                position_Q = position_T.repeat(1, 1, 1, q.size(2))
                position_A = position_T.repeat(1, 1, 1, a.size(2))
                # #print("position_T: {}".format(np.shape(position_T)))
                return position_Q, position_A

            def get_pos_simil(text_k, query_m, answer_n, pos_weight=None):
                """
                :param text_k:  (b,3,t,1,h)
                :param query_m: (b,3,1,q,h)
                :param answer_n:
                :param pos_weight:
                :return:
                """
                Q_km = F.cosine_similarity(text_k.repeat(1, 1, 1, q.size(2), 1),
                                           query_m.repeat(1, 1, t.size(2), 1, 1), dim=4)  # (b,3,t,q)
                A_kn = F.cosine_similarity(text_k.repeat(1, 1, 1, a.size(2), 1),
                                           answer_n.repeat(1, 1, t.size(2), 1, 1), dim=4)  # (b,3,t,a)
                if not pos_weight is None:
                    position_Q, position_A = get_position(pos_weight)
                    Q_km = Q_km * position_Q
                    A_kn = A_kn * position_A
                return Q_km, A_kn

            def get_M(Q_km, A_kn):
                # 公式（5）
                _MQ = torch.max(Q_km, dim=2)[0]  # (b,3,1,q)
                _MQ = _MQ.view(q.size(0), 3, -1)  # (b,3,q)
                MQ = _MQ.view(-1, q.size(2)).unsqueeze(1).bmm(w_m_q.view(-1, q.size(2), 1))  # (3b,1,q)(3b,q,1)->(3b,1,1)
                MQ = MQ.view(q.size(0), 3)  # (b,3) 3个维度上的MQ相等

                _MA = torch.max(A_kn, dim=2)[0]  # (b,3,1,a)
                _MA = _MA.view(a.size(0), 3, -1)  # (b,3,a)
                MA = _MA.view(-1, a.size(2)).unsqueeze(1).bmm(w_n_a.view(-1, a.size(2), 1))  # (3b,1,a)(3b,a,1) -> (3b,1,1)
                MA = MA.view(a.size(0), 3)  # (b,3) 3个维度上的MA不相等
                MA = F.softmax(MA, dim=1)

                MQ = F.dropout(MQ, self.drop_out)
                MA = F.dropout(MA, self.drop_out)

                return MQ, MA, MQ * MA

            #  4.2-(1)preparing
            tk = F.leaky_relu(self.W_B_t(t))  # （b,3,t,h)
            qm = F.leaky_relu(self.W_B_q(q))  # （b,3,q,h)
            an = F.leaky_relu(self.W_B_a(a))  # （b,3,a,h)
            # #print("tk: {}".format(np.shape(tk)))  # （b,3,t,h)
            # #print("an/qm: {}".format(np.shape(qm)))  # （b,3,a/q,h)
            #  4.2-(2)reshaping
            tk = tk.unsqueeze(3)
            # #print("tk un-squeezed(2):{}".format(np.shape(tk)))  # (b,3,t,1,h)
            qm = qm.unsqueeze(2)
            an = an.unsqueeze(2)
            # #print("an/qm un-squeezed(1):{}".format(np.shape(an)))  # (b,3,1,a/q,h)
            #  4.2-(3) weight vector for q and a
            w_m_q = self.V_q(q)  # (b,3,q,1)
            w_n_a = self.V_a(a)  # (b,3,a,1)
            # #print("w_n/m_a/q: {}".format(w_m_q.shape))

            # 4.2.1 Sentential
            cq_km, ca_kn = get_pos_simil(text_k=tk, query_m=qm, answer_n=an, pos_weight=None)  # fixme: slow
            # #print("ca_kn: {}".format(np.shape(ca_kn)))
            # #print("cq_km: {}".format(np.shape(cq_km)))

            Mq, Ma, Maq = get_M(cq_km, ca_kn)
            M_word = self.W_a1(Ma) + self.W_a2(Mq) + self.W_a3(Maq)
            # #print("--WbW/Sentential-- M_word: {}".format(np.shape(M_word)))

            # 4.2.2  Sequential Sliding Window
            sq_km, sa_kn = get_pos_simil(text_k=tk, query_m=qm, answer_n=an, pos_weight=self.position_t) # fixme: slow
            # #print("sa_kn: {}".format(np.shape(sa_kn)))
            # #print("sq_km: {}".format(np.shape(sq_km)))

            Mq, Ma, Maq = get_M(sq_km, sa_kn)
            M_sws = self.W_a4(Ma) + self.W_a5(Mq) + self.W_a6(Maq)
            # #print("--WbW/SWS-- M_word: {}".format(np.shape(M_sws)))

            # 4.2.3 Dependency Sliding Window
            # 根据fiedler向量的值给text重新排序
            tk = tk.view(tk.size(0), tk.size(1), tk.size(2), -1)
            # #print("tk: {}".format(np.shape(tk)))
            dep_idx, _ = padding([self.dep_info[int(id)] for id in ids], max_len=tk.size(2),limit_max=False)
            # #print("dep_mat: {}".format(np.shape(dep_idx)))

            tk_sort_init = np.zeros(shape=[tk.size(0), 3, t.size(2), tk.size(-1)])
            tk_sort = torch.FloatTensor(tk_sort_init)  # pytorch中只有float_tensor可以被优化
            for i in range(tk.size(0)):  # batch
                tk_i_sorted = tk[i, :, dep_idx[i], :]  # sort dim 2
                try:
                    tk_sort[i] = tk_i_sorted
                except Exception as e:
                    print(e)
                    print(tk_sort[i].shape, tk_i_sorted.shape)
            # #print("tk_sort: {}".format(tk_sort.shape))

            # 对重新排序的tk作sliding window处理
            tk = tk_sort.unsqueeze(3)
            sq_km, sa_kn = get_pos_simil(text_k=tk, query_m=qm, answer_n=an, pos_weight=self.position_t2)#  fixme: slow
            # #print("sa_kn: {}".format(np.shape(sa_kn)))
            # #print("sq_km: {}".format(np.shape(sq_km)))

            Mq, Ma, Maq = get_M(sq_km, sa_kn)
            M_swd = self.W_a7(Ma) + self.W_a8(Mq) + self.W_a9(Ma * Mq)
            # #print("--WbW/SWD-- M_word: {}".format(np.shape(M_sws)))

            aggregation = torch.cat([M_sem, M_word, M_sws, M_swd], dim=1)  # (b,12)
            # #print("aggregation: {}".format(aggregation.shape))

            # Layer4: Prediction Layer
            encoder_output = F.dropout(F.leaky_relu(self.MLP(aggregation)), self.drop_out)  # (b,2h)
            score = F.softmax(a_emb.bmm(encoder_output.unsqueeze(2)).squeeze(), 1)  # (b,3,2h) (b,2h,1)
            print("batch score: {}".format(Counter(score.argmax(1).data.numpy())[0] / self.opts["batch"]))
            # #print("batch score: {}".format(Counter(score.argmax(1).data.numpy())[0] / opts["batch"]))
            if not is_train:
                if is_argmax:
                    return score.argmax(1)
                else:
                    return score
            # loss = -torch.log(score[:, 0]).mean() # 原loss最大化score[0]的得分
            """we take the maximum over i so that  we  are  ranking  the  correct  answer  over  the  best-ranked
            incorrect answer (of which there are three) """
            correct = score[:, 0]
            m_score = torch.max(score, dim=1)[0]
            # #print(m_score.shape)
            u = 1.5  # score0 与 错误得分的间距
            margin = u + correct - m_score
            # #print(margin)
            zeros = torch.FloatTensor(np.zeros(shape=opts["batch"]))
            L = torch.max(zeros, margin)
            loss = L.mean()  # 最大化score0与错误选项的间距
            return loss
        except Exception as e:
            print(e)
            return 1



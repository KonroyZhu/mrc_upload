import argparse
import json
import pickle
import time

import numpy as np
import torch

from com.train_utils import train, test, train_main
from com.utils import shuffle_data, padding, pad_answer, get_model_parameters, load_t_d
from models.dcn import DCN
from models.dcn_mwan import DM
from models.mw_f_ori import Mw_f_ori
from models.mwan_f import MwAN_full
from models.phn import PHN

# from preprocess.get_emb import get_emb_mat
from models.qa_dcn import QA_DCN
# from models.qa_rnn import QA_RNN 过拟合严重
from models.qa_base import QA_Base
from models.qa_pos import QA_Pos
from models.qa_ps import QA_PS
from models.qa_self import QA_Self
from models.qa_selfmod import QA_SelfM
from models.qan import QAN

model_name = "trans"
desc = "64"
model_path = 'net/' + model_name + desc + '.pt'
record_path = 'net/' + model_name + desc + '.txt'


def get_emb_mat(id2v_path="../data/emb/id2v.pkl"):
    id2v = pickle.load(open(id2v_path, "rb"))
    id_list = sorted(list(id2v.keys()))
    embedding_matrix = [id2v[id] for id in id_list]
    return np.array(embedding_matrix)


if __name__ == '__main__':
    opts = json.load(open("models/config.json"))
    train_data, dev_data = load_t_d()
    # embedding_matrix = None
    embedding_matrix = torch.FloatTensor(get_emb_mat("data/emb/id2v.pkl"))/10

    # model = DCN(opts,embedding_matrix)
    # model=DM(opts,embedding_matrix)
    # model=PHN(opts,embedding_matrix) # 13406161
    # model=QA_Self(opts,embedding_matrix)
    # model=QA_PS(opts,embedding_matrix)
    # model=QA_SelfM(opts,embedding_matrix)
    # model=QA_DCN(opts,embedding_matrix)  # 16412800
    # model = QAN(opts, embedding_matrix)  # 16643073
    # model = QA_Base(opts, embedding_matrix)
    # model = QA_Pos(opts, embedding_matrix)
    # model =QA_RNN(opts,embedding_matrix)
    # model =Mw_f_ori(opts,embedding_matrix)
    model = MwAN_full(opts, embedding_matrix)  # 16821760
    print('Model total parameters:', get_model_parameters(model))
    if torch.cuda.is_available():
        model.cuda()
    # optimizer = torch.optim.Adamax(model.parameters())
    optimizer = torch.optim.Adam(lr=1e-3, betas=(0.8, 0.999), eps=1e-8, weight_decay=1e-7, params=model.parameters())

    best = 0.0
    best_epoch = 0
    train_main(model, train_data, dev_data, optimizer, best, best_epoch, model_path, record_path)  # FIXME: 把数据大小还原

import argparse
import json
import pickle
import time

import numpy as np
import torch

from com.train_utils import train, test, load_t_d, train_main
from com.utils import shuffle_data, padding, pad_answer, get_model_parameters
from models.dcn import DCN
from models.mw_f_ori import Mw_f_ori
from models.mwan_f import MwAN_full
from models.phn import PHN

# from preprocess.get_emb import get_emb_mat
from models.qa_dcn import QA_DCN
from models.qan import QAN

model_name = "mwan"
desc = "300"
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
    embedding_matrix = torch.FloatTensor(get_emb_mat("data/emb/id2v.pkl"))/5

    # model = DCN(opts,embedding_matrix)
    # model=PHN(opts,embedding_matrix) # 13406161
    # model=QA_DCN(opts,embedding_matrix)  # 16412800
    # model = QAN(opts, embedding_matrix)  # 16643073
    # model =Mw_f_ori(opts,embedding_matrix)
    model = MwAN_full(opts, embedding_matrix)  # 16821760
    print('Model total parameters:', get_model_parameters(model))
    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.Adamax(model.parameters())
    optimizer = torch.optim.Adam(lr=1e-4, betas=(0.8, 0.999), eps=1e-8, weight_decay=1e-7, params=model.parameters())

    best = 0.0
    best_epoch = 0
    train_main(model, train_data, dev_data, optimizer, best, best_epoch, model_path, record_path)

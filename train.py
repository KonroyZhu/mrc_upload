import argparse
import json
import pickle
import time

import torch

from com.train_utils import train, test, load_t_d, train_main
from com.utils import shuffle_data, padding, pad_answer, get_model_parameters
from models.dcn import DCN
from models.mw_f_ori import Mw_f_ori
from models.mwan_f import MwAN_full
from models.phn import PHN
from models.qa_net import QA_Net

# from preprocess.get_emb import get_emb_mat

model_name = "mwan_f"
desc = "d0.5"
model_path = 'net/' + model_name + desc + '.pt'
record_path = 'net/' + model_name + desc + '.txt'

if __name__ == '__main__':
    opts = json.load(open("models/config.json"))
    train_data, dev_data = load_t_d()
    embedding_matrix = None  # embedding_matrix=torch.FloatTensor(get_emb_mat("data/emb/id2v.pkl"))

    # model = DCN(opts,embedding_matrix)
    # model=PHN(opts,embedding_matrix) # 13406161
    # model=QA_Net(opts,embedding_matrix)  # 14844161
    # model =Mw_f_ori(opts,embedding_matrix)
    # model = MwAN_full(opts, embedding_matrix)  # 16821760
    print('Model total parameters:', get_model_parameters(model))
    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.Adamax(model.parameters())

    best = 0.0
    best_epoch = 0
    train_main(model, train_data, dev_data, optimizer, best, best_epoch, model_path, record_path)

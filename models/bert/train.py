import json
import pickle
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from com.utils import shuffle_data, padding
from models.bert.bert import BERT
from models.bert.bert_lm import BERTLM
from models.bert.data_utils import Data_utils
from models.bert.prepro import load_sentence_pair

def train_epoch(epoch,model,train_dt,dt_util, opt, best, best_epoch,batch_size=32):
    model.train()
    print("sentence pairs size:",np.shape(train_dt))
    data = shuffle_data(train_dt)
    data = train_dt
    total_loss = 0.0
    time_sum = 0.0
    for num, i in enumerate(range(0, len(data[:68]), batch_size)):
        time_start = time.time()
        ids = np.arange(start=i,stop=i + batch_size)
        batch_dt=[]
        for id in ids:
            batch_dt.append(dt_util.__getitem__(id))

        _inputs,_=padding([x[0] for x in batch_dt])
        _mask_lab,_=padding([x[1] for x in batch_dt])
        _seg_lab,_=padding([x[2] for x in batch_dt])
        _is_next=[x[3] for x in batch_dt]
        inputs,mask_lab,seg_lab,is_next=torch.LongTensor(_inputs),\
                                torch.LongTensor(_mask_lab),\
                                torch.LongTensor(_seg_lab),\
                                torch.LongTensor(_is_next) # ( b,t) (b,t) (b,t) (b,)
        criterion = nn.NLLLoss(ignore_index=0)
        next_sent_output, mask_lm_output = bert_lm.forward(inputs, seg_lab)  # (b,2) & (b,t,vocab)

        next_loss = criterion(next_sent_output, is_next)
        mask_loss = criterion(mask_lm_output.transpose(1, 2), mask_lab)
        loss = next_loss + mask_loss
        print(loss)

if __name__ == '__main__':
    opts = json.load(open("../config.json", "r"))
    sentence_pairs = load_sentence_pair(max_len=opts["p_len"])
    w2id = pickle.load(open("../../data/word2id.obj", "rb"))
    dt_u = Data_utils(sentence_pairs, w2id)
    # inp, t_label, seg_label,is_next_label = dt_u.__getitem__(0)
    bert = BERT(vocab_size=opts["vocab_size"])
    bert_lm = BERTLM(vocab_size=opts["vocab_size"], bert=bert)

    train_epoch(0,bert_lm,sentence_pairs,dt_u,None,0,0,32)



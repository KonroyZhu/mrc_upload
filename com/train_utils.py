import json
import pickle
import time

import numpy as np
import torch

from com.utils import padding, pad_answer, shuffle_data

opts = json.load(open("models/config.json"))


def train(epoch, net, train_dt, opt, best, best_epoch):
    net.train()
    data = shuffle_data(train_dt, 1)
    total_loss = 0.0
    time_sum = 0.0
    for num, i in enumerate(range(0, len(data), opts["batch"])):
        time_start = time.time()
        one = data[i:i + opts["batch"]]
        query, _ = padding([x[0] for x in one], max_len=opts["q_len"])
        passage, _ = padding([x[1] for x in one], max_len=opts["p_len"])
        answer = pad_answer([x[2] for x in one])
        ids = [x[3] for x in one]
        query, passage, answer, ids = torch.LongTensor(query), torch.LongTensor(passage), torch.LongTensor(answer), ids
        if torch.cuda.is_available():
            query = query.cuda()
            passage = passage.cuda()
            answer = answer.cuda()
        opt.zero_grad()
        loss = net([query, passage, answer, ids, True, True])
        loss.backward()
        total_loss += loss.item()
        opt.step()
        # 计时
        time_end = time.time()
        cost = (time_end - time_start)
        time_sum += cost
        if (num + 1) % opts["log_interval"] == 0:
            ts = str('%.2f' % time_sum)
            print(
                '|---epoch {:d} train error is {:f}  eclipse {:.2f}%  costing: {} best {} on epoch {}---|'.format(epoch,
                                                                                                                  total_loss /
                                                                                                                  opts[
                                                                                                                      "log_interval"],
                                                                                                                  i * 100.0 / len(
                                                                                                                      data),
                                                                                                                  ts + " s",
                                                                                                                  best,
                                                                                                                  best_epoch))
            time_sum = 0.0
            total_loss = 0


def test(net, valid_data):
    net.eval()
    r, a = 0.0, 0.0
    with torch.no_grad():
        for i in range(0, len(valid_data), opts["batch"]):
            print("{} in {}".format(i, len(valid_data)))
            one = valid_data[i:i + opts["batch"]]
            query, _ = padding([x[0] for x in one], max_len=opts["q_len"])
            passage, _ = padding([x[1] for x in one], max_len=opts["p_len"])
            answer = pad_answer([x[2] for x in one])
            ids = [x[3] for x in one]
            query, passage, answer, ids = torch.LongTensor(query), torch.LongTensor(passage), torch.LongTensor(
                answer), ids
            if torch.cuda.is_available():
                query = query.cuda()
                passage = passage.cuda()
                answer = answer.cuda()
            output = net([query, passage, answer, ids, False, True])
            r += torch.eq(output, 0).sum().item()
            a += len(one)
    return r * 100.0 / a


def train_main(model,train_data,dev_data,optimizer,best,best_epoch,model_path,record_path):
    for epoch in range(opts["epoch"]):
        train(epoch, model, train_data, optimizer, best, best_epoch)
        acc = test(net=model, valid_data=dev_data)
        if acc > best:
            best = acc
            best_epoch = epoch
            with open(model_path, 'wb') as f:
                torch.save(model, f)
            with open(record_path, 'w', encoding="utf-8") as f:
                f.write("best score: {} on epoch {}\n".format(best, best_epoch))

        print('epcoh {:d} dev acc is {:f}, best dev acc {:f}'.format(epoch, acc, best))

def ans_shuffle(answer):
    """
    将answer下标打乱，防止网络学习到正确下标为零的特征
    """
    batch =answer.shape[0]
    idx_s=np.random.randint(0,2,size=batch) # batch size
    idx_0=np.zeros_like(idx_s)
    placeholder=np.arange(0,batch) # batch size
    answer[placeholder,idx_s],answer[placeholder,idx_0] =answer[placeholder,idx_0],answer[placeholder,idx_s]
    return answer,idx_s
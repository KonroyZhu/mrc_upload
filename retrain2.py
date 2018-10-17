import argparse
import json
import pickle

import torch

from com.train_utils import test, train, load_t_d, train_main

model_name = "mwan_f"
desc = ""
model_path = 'net/' + model_name + desc + '.pt'
record_path = 'net/' + model_name + desc + '.txt'

if __name__ == '__main__':
    opts = json.load(open("models/config.json"))
    train_data, dev_data = load_t_d()
    with open(model_path, 'rb') as f:
        model = torch.load(f)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.Adamax(model.parameters(), lr=2e-4, weight_decay=1e-7)  # weight decay的作用是调节模型复杂度对损失函数的影响

    print("testing...")
    best = test(net=model, valid_data=dev_data)
    best_epoch = 0
    print("best: {}".format(best))
    print("training...")
    train_main(model, train_data, dev_data, optimizer, best, best_epoch, model_path, record_path)

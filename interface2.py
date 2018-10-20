import argparse
import json
import os
import pickle
import codecs

import numpy as np
import torch

from com.preprocess import transform_data_to_id
from com.utils import pad_answer, padding, pad_wrong_answer



def inference(model, data, md_name, dat_name, opts, is_argmax=True):
    model.eval()
    predictions = []
    id_prediction = {}
    with torch.no_grad():
        for i in range(0, len(data), opts["batch"]):
            print("{} in {}".format(i, len(data)))
            one = data[i:i + opts["batch"]]
            query, _ = padding([x[0] for x in one], max_len=50)
            passage, _ = padding([x[1] for x in one], max_len=300)
            answer = pad_answer([x[2] for x in one])
            str_words = [x[-1] for x in one]
            ids = [x[3] for x in one]
            answer = pad_wrong_answer(answer)
            query = torch.LongTensor(query)
            passage = torch.LongTensor(passage)
            # print(np.shape(answer))
            answer = torch.LongTensor(answer)
            if torch.cuda.is_available():
                query = query.cuda()
                passage = passage.cuda()
                answer = answer.cuda()
            output = model([query, passage, answer, ids, False, is_argmax])
            for q_id, prediction, candidates in zip(ids, output, str_words):
                if is_argmax:
                    id_prediction[q_id] = int(prediction)
                else:
                    prediction = prediction.cpu().numpy()
                    id_prediction[q_id] = list(prediction)
                prediction_answer = u''.join(candidates[np.argmax(prediction)])
                predictions.append(str(q_id) + '\t' + prediction_answer)
    outputs = u'\n'.join(predictions)
    with codecs.open("submit/" + md_name + "." + dat_name + ".txt", 'w', encoding='utf-8') as f:
        f.write(outputs)
    with open("pkl_records/" + md_name + "." + dat_name + ".pkl", "wb") as f:  # TODO: 更换pkl文件名称
        pickle.dump(id_prediction, f)
    print('done!')


def get_pkl(md_list, dt_name,opts,is_argmax=True):
    # raw_data = seg_data(args.data)
    with open("data/word2id.obj", 'rb') as f:
        word2id = pickle.load(f)
    raw_data = pickle.load(open("data/" + dt_name + "_seg.pkl", "rb"))  # TODO: 更改预测数据
    transformed_data = transform_data_to_id(raw_data, word2id)
    data = [x + [y[2]] for x, y in zip(transformed_data, raw_data)]
    data = sorted(data, key=lambda x: len(x[1]))
    print('test data size {:d}'.format(len(data)))

    for model_name in md_list:
        print("{} in [{}]".format(model_name," ".join(md_list)))
        model_path = "net/" + model_name + ".pt"
        with open(model_path, 'rb') as f:
            model = torch.load(f)
        if torch.cuda.is_available():
            model.cuda()
        inference(model, data, model_name, dt_name,opts,is_argmax)


if __name__ == '__main__':
    # TODO: 通过更换dt_name来更换数据集
    dt_name = "testa"
    # dt_name = "dev"
    # dt_name="train"
    opts = json.load(open("models/config.json"))
    f_list=os.listdir("net")
    md_list = [name.replace(".pt","") for name in f_list if ".pt" in name] # FIXME: dcn.pt should be .pt
    get_pkl(md_list,dt_name,opts,is_argmax=False)


import codecs
import os
import pickle

import numpy as np

from com.preprocess import transform_data_to_id
from com.utils import to_onehot, load_data


from ensm.score_on_dev import score_on_dt



def pkl_ens_pred_test(model_pkl_list, weight_list):
    model0_pkl = model_pkl_list[0]
    result = {}
    for j in range(len(list(model0_pkl.keys()))):
        k = list(model0_pkl.keys())[j]
        print("{} in {}".format(j, len(list(model0_pkl.keys()))))
        out = to_onehot(model0_pkl[k]) * weight_list[0]
        for i in range(1, len(model_pkl_list)):
            out += to_onehot(model_pkl_list[i][k]) * weight_list[i]
        decision = np.argmax(out)
        result[k] = decision
    return result


def pkl_pred(model_list, weight_list, query_ids,is_argmax):
    """ 预测key对应的答案下标"""
    decision=[]
    if is_argmax:
        for id in query_ids:
            model0_pkl = model_list[0]
            out = to_onehot(model0_pkl[id]) * weight_list[0]
            for i in range(1, len(model_list)):
                out += to_onehot(model_list[i][id]) * weight_list[i]
            decision.append(np.argmax(out))
    else:
        for id in query_ids:
            model0_pkl = model_list[0]
            out = np.array(model0_pkl[id]) * weight_list[0]
            for i in range(1, len(model_list)):
                out +=  np.array(model_list[i][id]) * weight_list[i]
            decision.append(np.argmax(out))
    return decision

def score_pred(id_prediction):
    hit=0.0
    for k in id_prediction.keys():
        if id_prediction[k] == 0:
            hit+=1
    print("score: {}".format(hit/len(list(id_prediction.keys()))))

# pred=pkl_ens_pred([dcn,mwan_o,mwan_f1,mwan_f0,mrc2_ori],
#                   [0.692467,0.6899, 0.693467,0.68967,0.67833])  # 0.70783
# pred=pkl_ens_pred([dcn,mwan_o,mwan_f1,mwan_f0,mrc2_ori,mrc2_ensemble],
#                   [0.692467,0.6899, 0.693467,0.68967,0.67833,0.6823])  # 0.7081


def interface_ori(md_list,weight_list,is_argmax=True):
    predictions = []
    id_prediction = {}
    for i in range(0, len(data), batch_size):
        print("{} in {}".format(i, len(data)))
        one = data[i:i + batch_size]
        str_words = [x[-1] for x in one]
        ids = [x[3] for x in one]
        output = pkl_pred(md_list,weight_list,ids,is_argmax)

        for q_id, prediction, candidates in zip(ids, output, str_words):
            id_prediction[q_id] = int(prediction)
            prediction_answer = u''.join(candidates[prediction])
            predictions.append(str(q_id) + '\t' + prediction_answer)
    outputs = u'\n'.join(predictions)
    with codecs.open("../submit/8th-submit.txt", 'w', encoding='utf-8') as f:
        f.write(outputs)
    return  id_prediction

# score_pred(id_prediction)
if __name__ == '__main__':
    # dt_name = "testa"
    dt_name = "dev"
    # dt_name="train"

    batch_size = 32
    data = load_data(data_path="../data/" + dt_name + "_seg.pkl", word2id_path='../data/word2id.obj')

    f_list=os.listdir("../pkl_records")
    md_pkl_list=[]
    for f in f_list:
        if dt_name in f:
             md_pkl_list.append(f)

    model_list=[]
    weight_list=[]
    for name in md_pkl_list:
        try:
            print(name,end=" ")
            dt_set = pickle.load(open("../pkl_records/"+name, "rb"))
            score=score_on_dt(dt_set,is_argmax=False)
            print(score)
            model_list.append(dt_set)
            weight_list.append(score)
        except Exception as e:
            print(name,e)
    """
    dcn.dev.pkl 0.6928333333333333
    dcnd0.5.dev.pkl 0.6920333333333333
    dcn_emb.dev.pkl 0.6734666666666667
    last_ensemble.dev.pkl 0.7059
    mwan_f.dev.pkl 0.6971666666666667
    mwan_fd0.5.dev.pkl 0.6932333333333334
    mw_f_ori.dev.pkl 0.6980333333333333"""
    weight_list = [0.6928333333333333,0.6920333333333333,0.6734666666666667,0.7059, 0.6971666666666667,0.6932333333333334,0.6980333333333333]
    id_pred=interface_ori(model_list,weight_list,is_argmax=True) # argmax_False is lower
    pickle.dump(id_pred,open("../pkl_records/ensemble."+dt_name+".pkl","wb"))
    print(score_on_dt(id_pred,is_argmax=False))

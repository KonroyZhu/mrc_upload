import pickle

import numpy as np

dt_set=pickle.load(open("../pkl_records/last_ensemble.dev.pkl","rb"))  # 0.499

def score_on_dt(dt_set,is_argmax=True):
    hit=0.0
    for k in dt_set.keys():
        ans=dt_set[k]
        if is_argmax:
            ans = np.argmax(ans)
        if ans==0:
            hit+=1
    return hit / len(list(dt_set.keys()))

if __name__ == '__main__':
    print(score_on_dt(dt_set))
import pickle

import numpy as np


def score_on_dt(dt_set, need_argmax=True):
    hit = 0.0
    for k in dt_set.keys():
        ans = dt_set[k]
        if need_argmax:
            ans = np.argmax(ans)
        if ans == 0:
            hit += 1
    return hit / len(list(dt_set.keys()))


if __name__ == '__main__':
    dt_set = pickle.load(open("../pkl_records/1/ensemble01.dev.pkl", "rb"))  # 0.499
    print(score_on_dt(dt_set, need_argmax=False))

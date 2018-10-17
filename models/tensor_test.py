import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

t=torch.FloatTensor(np.random.rand(32,50,128))
print(t.shape)
tp=F.pad(t,(0,0,0,20))  # 0,0: 在最后两个维度前后补零 0,20: 在倒数第二个维度前面补0后面补20
print(tp.shape)
print(tp)

# tt=torch.cuda.FloatTensor([0,1,2])
# print(tt)
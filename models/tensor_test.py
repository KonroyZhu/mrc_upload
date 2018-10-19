import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.qa_net import DepthwiseSeparableConv

""" padding test
t=torch.FloatTensor(np.random.rand(32,50,128))
print(t.shape)
tp=F.pad(t,(0,0,0,20))  # 0,0: 在最后两个维度前后补零 0,20: 在倒数第二个维度前面补0后面补20
print(tp.shape)
print(tp)
"""
tt3 = torch.FloatTensor(np.random.rand(96,128,3))
tt1 = torch.FloatTensor(np.random.rand(96,128,1))
p_conv_dws = DepthwiseSeparableConv(in_ch=128, out_ch=128, k=5)
print("tt3",p_conv_dws(tt3).shape)
print("tt1",p_conv_dws(tt1).shape)
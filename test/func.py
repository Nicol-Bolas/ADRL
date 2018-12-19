import torch
import torch.nn.functional as F
import numpy as np
import random
import torch.nn as nn
# from FrameEvaluation.FrameKeeper import FrameKeeper as FK
from FrameEvaluation.Dataset import ADRLDataset as ADRLds
import mytool.fileprocess as fp


input1 = torch.randn(100, 128)
input2 = torch.randn(100, 128)
output = F.cosine_similarity(input1, input2)

a = torch.arange(9, dtype= torch.float) - 4
# print(torch.sum(torch.pow(a,2)))
# print(a)
# print(torch.norm(a,2))

print(np.ones(5,dtype=int).tolist())
print(list(range(0,5)))
a=torch.Tensor([1,2])
b=torch.Tensor([2,4])
loss=nn.MSELoss()
print(loss(a,b))

a_drop_num=np.random.randint(1,5)
print(a_drop_num)

a_drop_list=random.sample(range(0,5),4)
print(a_drop_list)

random_select=random.random()>0.6
print(random_select)

# frames,frame_names=FK.FrameKeeper.load_frames_from_dir('E:/work/python/pytorch/ADRL/Paul_Kariya/5')
# frame_feas=FK.FrameKeeper.load_feas_from_dir(frame_names,r'E:\work\python\pytorch\ytf_feat_part.txt')

# fk_a=FK('E:/work/python/pytorch/ADRL/Talisa_Soto/1',r'E:\work\python\pytorch\ytf_feat_part.txt')
# fk_a.drop_frames([0,2,4,6,8,10,12,14])
# print(len(fk_a.remaining_frames()))
# fk_a.drop_single_frame(2)
# print(len(fk_a.remaining_frames()))
# fk_a.reset()
# print(len(fk_a.remaining_frames()))

# label_file='E:/work/python/pytorch/ADRL/label.txt'
# label_list=fp.ReadAll(label_file,True)
# ds=ADRLds(label_list)
# print(len(ds))
# print(ds[0])

a=torch.tensor([[1., 1.], [2., 2.], [3., 3.]],requires_grad=True)
b=torch.tensor([[11., 11.], [22., 22.], [33., 33.]],requires_grad=True)
c=torch.tensor([[111., 111.], [222., 222.], [333., 333.]],requires_grad=True)
d=torch.tensor([[1111., 1111.], [2222., 2222.], [3333., 3333.]],requires_grad=True)

e=torch.stack( [a,b,c,d] ,dim = 0)

mlist=[1,2,3,4,5,6,7,8,9,10]
for i in range(10):
    if i%2==0:
        mlist.pop(i)

print(mlist)


print('finish')
import torch
from torchvision import transforms

import FrameEvaluation.utils as FEUtil
from FrameEvaluation.Dataset import ADRLDataset as ADRLds
from FrameEvaluation.QNet import QNet
from FrameEvaluation.Trainer import ADRLTrainer as Trainer

from Utils.checkpoint import CheckPoint
from Utils.mylog import MyLogStream
import Utils.fileoperator as fp

v_dim=1024
max_state_num=8
# greedy_factor=[(1.0,4),(0.8,12),(0.6,20),(0.4,28),(0.2,36),(0.1,40)]
greedy_factor=[(1.0,20),(0.9,40),(0.8,80),(0.7,100),(0.6,120),(0.5,140),(0.4,160),(0.3,180),(0.2,200),(0.1,220)]
discount_factor=0.98
frame_width=96
frame_height=112
max_epoch=3


label_file='E:/work/python/pytorch/ADRL_data/new_label.txt'
# feas_path='E:/work/python/pytorch/ytf_feat_part.txt'
frame_root_dir='E:/work/python/pytorch/ADRL_data/ytf_frame/'
save_path='E:/work/python/pytorch/ADRL_data/trained-model/'
model_name='adrl4'

label_list=fp.read_file(label_file,True)
logger=MyLogStream(save_path+model_name+'_log.txt')
checkpoint = CheckPoint(save_path,model_name)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

ds=ADRLds(label_list,frame_root_dir)
train_loader = torch.utils.data.DataLoader(ds,batch_size=4)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

train_model=QNet(v_dim).to(device)
start_lr=1e-6
optimizer = torch.optim.RMSprop(train_model.parameters(), lr=start_lr,weight_decay=0.9)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150], gamma=0.5)
scheduler = None

trainer=Trainer(start_lr, train_loader, train_model, optimizer, scheduler, device,max_state_num,greedy_factor,discount_factor,frame_width,frame_height,transform=transform,logger=logger,feat_dim=v_dim)

print('start to train....')
for epoch in range(max_epoch):
    trainer.train_epoch()
    checkpoint.save_model(train_model, index=epoch + 1)

print('finish')
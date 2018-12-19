import torch.nn as nn
import torch.nn.functional as F
import torch


class ADRLLoss:
    def __init__(self, device):
        self.loss=nn.MSELoss().to(device)

    def compute(self,q_score,y):
        return self.loss(q_score,y)


class QNet(nn.Module):
    def __init__(self,v_dim):
        super(QNet,self).__init__()
        self.v_dim=v_dim
        self.pre_layer=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,stride=1,kernel_size=9),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=1),
            nn.Conv2d(in_channels=16,out_channels=32,stride=1,kernel_size=4),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32,out_channels=64,stride=1,kernel_size=3),
            nn.PReLU(),
        )
        self.first_order_feature= nn.Sequential(
            nn.Linear(v_dim, 64),
            nn.Tanh(),
        )
        self.second_order_feature = nn.Sequential(
            nn.Linear(v_dim, 64),
            nn.Tanh(),
        )
        self.fc1=nn.Linear(64*23*19,64)
        self.fc2=nn.Linear(64*5,64)
        self.fc3=nn.Linear(64,1)

    def forward(self,drop_frame_img,drop_frame_fea,mean_state_video_a,mean_state_video_b,var_state_video_a,var_state_video_b):
        if len(drop_frame_img.shape)<4:
            drop_frame_img=drop_frame_img.unsqueeze(0)
        qnet_fea=self.pre_layer(drop_frame_img)
        qnet_fea=qnet_fea.view(qnet_fea.size(0),-1)
        qnet_fea=self.fc1(qnet_fea)

        video_a_1_fea=self.first_order_feature(mean_state_video_a-drop_frame_fea).unsqueeze(0)
        video_b_1_fea = self.first_order_feature(mean_state_video_b - drop_frame_fea).unsqueeze(0)
        video_a_2_fea=self.second_order_feature(var_state_video_a).unsqueeze(0)
        video_b_2_fea = self.second_order_feature(var_state_video_b).unsqueeze(0)

        qnet_fea=torch.cat((qnet_fea,video_a_1_fea,video_a_2_fea,video_b_1_fea,video_b_2_fea),1)
        qnet_fea=self.fc2(qnet_fea)
        q_score=self.fc3(qnet_fea)

        return q_score
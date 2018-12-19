import torch
import FrameEvaluation.QNet as QNet
import FrameEvaluation.FrameKeeper as FK
import numpy as np
import random
import FrameEvaluation.utils as FEUtil
import copy


class ADRLTrainer:

    def __init__(self,lr, train_loader, model, optimizer, scheduler, device,max_state_num,greedy_factor,discount_factor,frame_width,frame_height,feas_path=None,transform = None,logger=None,feat_dim=512):
        self.lr = lr
        self.train_loader = train_loader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.epoch = 1
        self.loss=QNet.ADRLLoss(device)
        self.max_state_num=max_state_num

        self.greedy_factor=greedy_factor
        self.greedy_idx=0
        self.current_gf=greedy_factor[0][0]
        self.change_batch=greedy_factor[0][1]
        self.batch_num=0

        self.discount_factor=discount_factor
        self.std_w = frame_width
        self.std_h = frame_height
        self.transform = transform
        self.feas_path=feas_path
        self.feat_dim=feat_dim
        self.logger = logger

    def change_greedy(self):
        self.greedy_idx+=1
        if self.greedy_idx<len(self.greedy_factor):
            self.current_gf=self.greedy_factor[self.greedy_idx][0]
            self.change_batch=self.greedy_factor[self.greedy_idx][1]
        else:
            self.change_batch=-1

    def train_epoch(self):
        if self.scheduler is not None:
            self.scheduler.step()
        self.model.train()

        final_loss_total = 0
        data_num = 0

        for idx,(video_paths,labels) in enumerate(self.train_loader):

            y_score=[]
            q_score=[]

            for batch_idx in range(labels.shape[0]):

                video_a_path=video_paths[0][batch_idx]
                video_b_path=video_paths[1][batch_idx]
                label=labels[batch_idx]

                keeper_a=FK.FrameKeeper(video_a_path,self.std_w,self.std_h,transform=self.transform,feat_dim=self.feat_dim,fea_path=self.feas_path)
                keeper_b = FK.FrameKeeper(video_b_path,self.std_w,self.std_h,transform=self.transform,feat_dim=self.feat_dim,fea_path=self.feas_path)

                for i in range(self.max_state_num):
                    a_drop_num=random.randint(1,keeper_a.video_len-3)
                    b_drop_num =random.randint(1, keeper_b.video_len - 3)
                    a_drop_list=random.sample(range(0,keeper_a.video_len),a_drop_num)
                    b_drop_list = random.sample(range(0, keeper_b.video_len), b_drop_num)

                    keeper_a.drop_frames(a_drop_list)
                    keeper_b.drop_frames(b_drop_list)

                    random_drop=random.random()>self.current_gf
                    dropped_frame_fea = None
                    dropped_frame = None

                    a_remaining_befor_drop=copy.deepcopy(keeper_a.remaining_frame_feas())
                    b_remaining_befor_drop = copy.deepcopy(keeper_b.remaining_frame_feas())

                    # take drop action by greedy
                    if random_drop:
                        if random.random() > 0.5:
                            drop_idx = random.randint(0, keeper_a.remaining_length()-1)
                            dropped_frame_fea, dropped_frame = keeper_a.drop_single_frame(drop_idx)
                        else:
                            drop_idx = random.randint(0, keeper_b.remaining_length()-1)
                            dropped_frame_fea, dropped_frame = keeper_b.drop_single_frame(drop_idx)
                    else:
                        q_max_score, q_max_idx, q_video=FEUtil.select_max_qScore(keeper_a.remaining_frames(),keeper_a.remaining_frame_feas(),keeper_b.remaining_frames(),keeper_b.remaining_frame_feas(),self.model)

                        if q_video==0:
                            dropped_frame_fea, dropped_frame = keeper_a.drop_single_frame(q_max_idx)
                        else:
                            dropped_frame_fea, dropped_frame = keeper_b.drop_single_frame(q_max_idx)

                    reward=FEUtil.action_reward(label,a_remaining_befor_drop,b_remaining_befor_drop,keeper_a.remaining_frame_feas(),keeper_b.remaining_frame_feas())

                    # calculate new state and variance after action
                    new_mean_state_video_a = FEUtil.mean_state_of_video(keeper_a.remaining_frame_feas())
                    new_mean_state_video_b = FEUtil.mean_state_of_video(keeper_b.remaining_frame_feas())

                    new_var_state_video_a = FEUtil.var_state_of_video(keeper_a.remaining_frame_feas())
                    new_var_state_video_b = FEUtil.var_state_of_video(keeper_b.remaining_frame_feas())

                    temp_q_score=self.model(dropped_frame,dropped_frame_fea,new_mean_state_video_a, new_mean_state_video_b,new_var_state_video_a, new_var_state_video_b)
                    q_score.append(temp_q_score)

                    if reward<0:
                        temp_y_score=reward
                    else:
                        new_q_max_score, new_q_max_idx, new_q_video = FEUtil.select_max_qScore(keeper_a.remaining_frames(),keeper_a.remaining_frame_feas(), keeper_b.remaining_frames(),keeper_b.remaining_frame_feas(), self.model)
                        temp_y_score=reward+self.discount_factor*new_q_max_score.double()
                    y_score.append(temp_y_score.squeeze())

                    keeper_a.reset()
                    keeper_b.reset()

            self.batch_num+=1
            if self.change_batch>0 and self.batch_num>=self.change_batch:
                self.change_greedy()

            q_score=torch.stack(q_score ,dim = 0).double().squeeze()
            y_score=torch.stack(y_score ,dim = 0).double()
            final_loss=self.loss.compute(q_score,y_score)

            self.optimizer.zero_grad()
            final_loss.backward()
            self.optimizer.step()

            final_loss_total+=final_loss.item()*q_score.shape[0]
            data_num+=q_score.shape[0]

            print('train epoch: {} [{}/{}]\tloss: {:.4f}'.format(self.epoch, idx+1, len(self.train_loader), final_loss.item()))

            if self.logger is not None:
                batch_log = 'train epoch: {} [{}/{}]\tloss: {:.4f}'.format(self.epoch, idx+1, len(self.train_loader), final_loss.item())
                self.logger.write_line(batch_log)

        if self.logger is not None:
            epoch_log="epoch:{},avg loss:{:.4f}".format(self.epoch,final_loss_total/data_num)
            self.logger.write_line(epoch_log)

        self.epoch += 1
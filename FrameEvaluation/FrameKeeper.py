import numpy as np
import cv2
import os
from PIL import Image
import Utils.fileoperator as fp
import torch
import copy
import FrameEvaluation.utils as FEUtil

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class FrameKeeper:
    def __init__(self,video_path,frame_width,frame_height,feat_dim=512,transform = None,interval=1,fea_path=None,device=None):
        # self.frames:list
        frames, frame_names = FrameKeeper.load_frames_from_dir(video_path)
        if fea_path is None:
            fea_path=video_path+'/feat.txt'
        frame_feas = FrameKeeper.load_feas_from_dir(frame_names, fea_path,feat_dim=feat_dim,device=device)
        cleaned_frames,cleaned_frame_names,cleaned_frame_feas=FrameKeeper.clean_unpaired_data(frames,frame_names,frame_feas)

        # self.frames=FrameKeeper.split_video_to_frames(video_path,interval)
        # self.frame_feas=FrameKeeper.extract_features(self.frames)
        self.frames =cleaned_frames
        self.frame_feas =cleaned_frame_feas
        self.frame_names=cleaned_frame_names
        self.video_len=len(self.frames)
        self.attention=np.ones(self.video_len,dtype=int).tolist()
        self.exist_frame_feas=copy.deepcopy(self.frame_feas)
        self.frame_fea_index=list(range(0,self.video_len))
        self.std_w=frame_width
        self.std_h=frame_height
        self.transform = transform
        self.frames_resize()
        self.frames_trans(device)

    def frames_resize(self):
        for i in range(self.video_len):
            frame=self.frames[i]
            if frame.width!=self.std_w or frame.height!=self.std_h:
                frame=frame.resize((self.std_w, self.std_h), Image.ANTIALIAS)
                self.frames[i]=frame

    def frames_trans(self,device=None):
        if self.transform is not None:
            for i in range(self.video_len):
                frame = self.frames[i]
                frame=self.transform(frame)
                if device is not None:
                    frame=frame.to(device)
                self.frames[i]=frame

    def drop_single_frame(self,idx):
        if idx>len(self.exist_frame_feas)-1 or idx<0:
            return None
        dropped_frame_fea=self.exist_frame_feas.pop(idx)
        frame_idx=self.frame_fea_index.pop(idx)
        self.attention[frame_idx]=0
        dropped_frame =self.frames[frame_idx]
        self.attention[frame_idx]=0
        return dropped_frame_fea,dropped_frame

    def drop_frames(self,idxs):
        if len(idxs)<len(self.exist_frame_feas):
            temp_frame_feas=[]
            temp_idxs=[]
            for idx in self.frame_fea_index:
                if idx not in idxs:
                    temp_frame_feas.append(self.exist_frame_feas[idx])
                    temp_idxs.append(self.frame_fea_index[idx])
                else:
                    self.attention[self.frame_fea_index[idx]]=0
            self.exist_frame_feas=copy.deepcopy(temp_frame_feas)
            self.frame_fea_index=copy.deepcopy(temp_idxs)

    def remaining_frame_feas(self):
        return self.exist_frame_feas

    def remaining_frames(self):
        r_frames=[]
        for idx in self.frame_fea_index:
            r_frames.append(self.frames[idx])
        return r_frames

    def remaining_frame_names(self):
        r_names=[]
        for idx in self.frame_fea_index:
            r_names.append(self.frame_names[idx])
        return r_names

    def remaining_length(self):
        return len(self.exist_frame_feas)

    def reset(self):
        self.exist_frame_feas=copy.deepcopy(self.frame_feas)
        self.frame_fea_index=list(range(0,self.video_len))
        self.attention = np.ones(self.video_len, dtype=int).tolist()

    @classmethod
    def split_video_to_frames(cls,video_path,interval):
        video_frames=[]
        cap=cv2.VideoCapture(video_path)
        success=False
        frame_idx=0
        if cap.isOpened():
            success=True

        while(success):
            success,frame=cap.read()
            if frame_idx%interval==0:
                video_frames.append(frame)
            frame_idx+=1
        return video_frames

    @classmethod
    def extract_features(cls,frames):
        features=[]
        # test
        features=list(range(0,len(frames)))
        return features

    @classmethod
    def load_frames_from_dir(cls,frames_dir):
        frames=[]
        frame_names=[]
        for file in os.listdir(frames_dir):
            ext=os.path.splitext(file)[1]
            if ext=='.jpg' or ext=='.bmp':
                frames.append(pil_loader(frames_dir+'/'+file))
                splits=(frames_dir+'/'+file).split('/')
                frame_names.append(splits[-3]+'/'+splits[-2]+'/'+splits[-1])
        return frames,frame_names

    @classmethod
    def load_feas_from_dir(cls, frame_names,feas_path,feat_dim=512,device=None):
        feas = list(range(len(frame_names)))
        for i in range(len(frame_names)):
            feas[i]=None
        data_list=fp.read_file(feas_path)
        for data in data_list:
            splits=data.split('\t')
            name=splits[0]
            feat = np.zeros((feat_dim,), dtype=np.float32)
            for i in range(feat_dim):
                try:
                    feat[i] = splits[i + 1]
                except ValueError:
                    feat[i] = 1.0
            # feat=torch.from_numpy(feat).requires_grad_()
            feat = torch.from_numpy(feat)
            if device is not None:
                feat=feat.to(device)
            for i in range(len(frame_names)):
                if frame_names[i]==name:
                    feas[i]=feat
        return feas

    @classmethod
    def clean_unpaired_data(cls,frames,frame_names,frame_feas):
        new_frames=[]
        new_frame_feas=[]
        new_frame_names=[]
        for i in range(len(frame_feas)):
            if frame_feas[i] is not None:
                # new_frames.append(copy.deepcopy(frames[i]))
                # new_frame_feas.append(copy.deepcopy(frame_feas[i]))
                new_frames.append(frames[i])
                new_frame_feas.append(frame_feas[i])
                new_frame_names.append(frame_names[i])
        return new_frames,new_frame_names,new_frame_feas

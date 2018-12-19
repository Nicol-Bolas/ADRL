from FrameEvaluation.FrameKeeper import FrameKeeper
import FrameEvaluation.utils as FEUtil


class FrameDroppper:
    def __init__(self,video_a_path,video_b_path,threshold,dqn_model,frame_width,frame_height,transform = None,fea_path=None,feat_dim=512):
        self.keeper_a=FrameKeeper(video_a_path,frame_width,frame_height,transform=transform,fea_path=fea_path,feat_dim=feat_dim)
        self.keeper_b = FrameKeeper(video_b_path, frame_width, frame_height, transform=transform, fea_path=fea_path,feat_dim=feat_dim)
        self.dqn_model=dqn_model
        self.threshold=threshold
        self.step=0

    def drop_frame(self,video_id,frame_id):
        if video_id == 0:
            self.keeper_a.drop_single_frame(frame_id)
        else:
            self.keeper_b.drop_single_frame(frame_id)

    def get_attention(self):
        drop=True
        if self.keeper_a.remaining_length()<=self.threshold or self.keeper_b.remaining_length()<=self.threshold:
            drop=False
        else:
            q_max_score, q_max_idx, q_video = FEUtil.select_max_qScore(self.keeper_a.remaining_frames(),self.keeper_a.remaining_frame_feas(),self.keeper_b.remaining_frames(),self.keeper_b.remaining_frame_feas(),self.dqn_model)
            self.drop_frame(q_video,q_max_idx)
            self.step+=1

        while(drop):
            q_max_score, q_max_idx, q_video=FEUtil.select_max_qScore(self.keeper_a.remaining_frames(),self.keeper_a.remaining_frame_feas(),self.keeper_b.remaining_frames(),self.keeper_b.remaining_frame_feas(), self.dqn_model)
            if q_max_score<0 or self.keeper_a.remaining_length()<=self.threshold or self.keeper_b.remaining_length()<=self.threshold:
                drop=False
            else:
                self.drop_frame(q_video, q_max_idx)
                self.step += 1

        return self.step,self.keeper_a.remaining_frames(),self.keeper_b.remaining_frames()

    def get_attention_ori(self):
        drop=True
        if self.keeper_a.remaining_length()<=self.threshold or self.keeper_b.remaining_length()<=self.threshold:
            drop=False

        while(drop):
            q_max_score, q_max_idx, q_video=FEUtil.select_max_qScore(self.keeper_a.remaining_frames(),self.keeper_a.remaining_frame_feas(),self.keeper_b.remaining_frames(),self.keeper_b.remaining_frame_feas(), self.dqn_model)

            self.drop_frame(q_video, q_max_idx)

            q_max_score, q_max_idx, q_video = FEUtil.select_max_qScore(self.keeper_a.remaining_frames(),self.keeper_a.remaining_frame_feas(),self.keeper_b.remaining_frames(),self.keeper_b.remaining_frame_feas(),self.dqn_model)

            if q_max_score<0 or self.keeper_a.remaining_length()<=self.threshold or self.keeper_b.remaining_length()<=self.threshold:
                drop=False
            else:
                self.step += 1

        return self.step,self.keeper_a.remaining_frames(),self.keeper_b.remaining_frames()

    def get_similarity(self):
        return FEUtil.similarity_of_videos(self.keeper_a.remaining_frame_feas(),self.keeper_b.remaining_frame_feas()).item()

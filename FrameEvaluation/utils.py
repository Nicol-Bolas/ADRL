import torch
import copy


def mean_state_of_video(remaining_video_frame_feas):
    final_fea=copy.deepcopy(remaining_video_frame_feas[0])
    for i in range(1,len(remaining_video_frame_feas)):
        final_fea+=remaining_video_frame_feas[i]
    final_fea=final_fea/len(remaining_video_frame_feas)
    return final_fea


def var_state_of_video(remaining_video_frame_feas):
    aver_fea=mean_state_of_video(remaining_video_frame_feas)
    final_fea=aver_fea-aver_fea
    for frame_fea in remaining_video_frame_feas:
        final_fea+=torch.pow((frame_fea-aver_fea),2)
    final_fea=final_fea/len(remaining_video_frame_feas)
    return final_fea


def similarity_of_videos(remaining_video_a_feas,remaining_video_b_feas):
    mean_state_a=mean_state_of_video(remaining_video_a_feas)
    mean_state_b = mean_state_of_video(remaining_video_b_feas)
    mean_state_a=mean_state_a/torch.norm(mean_state_a)
    mean_state_b=mean_state_b/torch.norm(mean_state_b)

    return torch.sum(mean_state_a*mean_state_b)
    # return torch.from_numpy(mean_state_a.numpy().dot(mean_state_b.numpy()))
    # return mean_state_a*mean_state_b


def action_reward(label,remaining_video_a_feas_old,remaining_video_b_feas_old,remaining_video_a_feas_new,remaining_video_b_feas_new):
    similarity_old=similarity_of_videos(remaining_video_a_feas_old,remaining_video_b_feas_old)
    similarity_new=similarity_of_videos(remaining_video_a_feas_new,remaining_video_b_feas_new)
    return label*(similarity_new-similarity_old).double()


def select_max_qScore(video_a_frames,video_a_frame_feas,video_b_frames,video_b_frame_feas,model):
    q_max_score = 0
    q_max_idx = 0
    q_video = 0

    # loop all remaining frames in video_a to find max q_score while video_b do not change
    # temp_frames = keeper_a.remaining_frames()
    for i in range(len(video_a_frame_feas)):
        temp_frame_feas = copy.deepcopy(video_a_frame_feas)
        temp_drop_frame_fea = temp_frame_feas.pop(i)
        mean_state_video_a = mean_state_of_video(temp_frame_feas)
        mean_state_video_b = mean_state_of_video(video_b_frame_feas)
        var_state_video_a = var_state_of_video(temp_frame_feas)
        var_state_video_b = var_state_of_video(video_b_frame_feas)

        q_score_temp = model(video_a_frames[i], temp_drop_frame_fea, mean_state_video_a, mean_state_video_b, var_state_video_a, var_state_video_b)
        if i == 0:
            q_max_score = q_score_temp
        else:
            if q_max_score < q_score_temp:
                q_max_score = q_score_temp
                q_max_idx = i

    # loop all remaining frames in video_b to find max q_score while video_a do not change
    for i in range(len(video_b_frame_feas)):
        temp_frame_feas = copy.deepcopy(video_b_frame_feas)
        temp_drop_frame_fea = temp_frame_feas.pop(i)

        mean_state_video_a = mean_state_of_video(video_a_frame_feas)
        mean_state_video_b = mean_state_of_video(temp_frame_feas)

        var_state_video_a = var_state_of_video(video_a_frame_feas)
        var_state_video_b = var_state_of_video(temp_frame_feas)

        q_score_temp = model(video_b_frames[i], temp_drop_frame_fea, mean_state_video_a, mean_state_video_b,
var_state_video_a, var_state_video_b)
        if q_max_score < q_score_temp:
            q_max_score = q_score_temp
            q_max_idx = i
            q_video = 1

    return q_max_score,q_max_idx,q_video


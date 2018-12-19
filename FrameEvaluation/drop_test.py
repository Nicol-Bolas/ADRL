from FrameEvaluation.FrameDrop import FrameDroppper
from FrameEvaluation.QNet import QNet
from Utils.checkpoint import CheckPoint

from torchvision import transforms

model_path=r'E:\work\python\pytorch\ADRL_data\trained-model\check_point_adrl3\model_010.pth'
video_a_path='E:/work/python/pytorch/ADRL_data/ytf_frame/Andy_Dick/1'
video_b_path='E:/work/python/pytorch/ADRL_data/ytf_frame/Zulfiqar_Ahmed/1'
v_dim=1024
frame_width=96
frame_height=112
threshold=5

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dqn_model=QNet(v_dim)
dqn_model=CheckPoint.load_model_from_path(dqn_model,model_path)
dqn_model.eval()

dropper=FrameDroppper(video_a_path,video_b_path,threshold,dqn_model,frame_width,frame_height,transform=transform,feat_dim=v_dim)

step,a_remaining_frames,b_remaining_frames=dropper.get_attention()

print(dropper.keeper_b.remaining_frame_names())
print(dropper.keeper_a.remaining_frame_names())
print(dropper.get_similarity())

print('finish')
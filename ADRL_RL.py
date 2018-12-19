import torch
import torch.nn as nn
import argparse, os, random, cv2, sys, time, math, copy
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
#from interval import Interval
from torch.autograd import Variable
#from torchvision.utils import save_image
from scipy import spatial
from collections import namedtuple
#from itertools import count
from concurrent.futures import ThreadPoolExecutor, as_completed

# ADRL_Util.py
import ADRL_Util

#sys.path.append('/home/yezilong/my_model/AAA')
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

class Q_Network(nn.Module):
    def __init__(self, h_feat_dim=1024):
        super(Q_Network, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),    #112*96
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),                               #56*48
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  #28*24
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),                               #14*12
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), #14*12
            nn.PReLU(),
        )

        self.L1 = nn.Linear(h_feat_dim, 64)
        self.L2 = nn.Linear(h_feat_dim, 64)
        self.final_fc = nn.Linear(21760,1)

    def forward(self, img, pa_feat, pb_feat, var_a_feat, var_b_feat, droping_h_feat):  # img (batch, 3, 112, 96)  other (batch, h_feat_dim)
        img = self.model(img)

        # froward L1 L2
        a1_order, b1_order = torch.tanh(self.L1(pa_feat - droping_h_feat)), torch.tanh(self.L1(pb_feat - droping_h_feat))
        a2_order, b2_order = torch.tanh(self.L2(var_a_feat)), torch.tanh(self.L2(var_b_feat))
        # combine v_i
        v_i = torch.cat([a1_order, a2_order, b1_order, b2_order], 1)

        # count Q_i
        concat_feat = torch.cat([img.view(img.size(0), -1), v_i], 1)
        Q_i = self.final_fc(concat_feat)
        return Q_i


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def get_Lately_transition(self): # 获取最新进入memory的transition, 如果不存在就返回None
        if len(self.memory) == 0: return None
        position = (self.position - 1) % self.capacity
        return self.memory[position]

def save_model(args, model, epoch):
    args.epoch = epoch
    args.arch_type = 'ADRL_RL'
    utils.save_epoch(args, model)


###########################################################################################
frame_h_feat_map = ADRL_Util.get_frame_h_feat_map()
#frame_h_feat_map = {}
video_frames_map = ADRL_Util.get_video_frames_map()
train_lines, test_lines  = ADRL_Util.split_train_test(test_split = 9, split_file = '/home/yezilong/my_model/valid/splits_no_header.txt')
pair_steps_map = {}  # dynamic init

BATCH_SIZE = 32
GAMMA = 0.98
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 8

act_batch = 128
memory_capacity = 128
Video_Group = 16
Memory_Batch = 8

YTF_Root_Dir = '/home/yezilong/dataset/YouTubeFaces/112X96'
ytf_r = 3
use_cuda = True


State = namedtuple('State',
                        ('h_list', 'v_a', 'v_b', 'label', 'split_index', 'remain_a', 'remain_b'))

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state_end', 'reward', 'Q_i', 'cache_next_Qi'))  # state must copy note： state的h_list 需要在新建下个状态时最新赋值

executor = ThreadPoolExecutor(max_workers=16)

policy_net = Q_Network()
# resume
#checkpoint = torch.load('/home/yezilong/my_model/AAA/result/checkpoint/u_ADRL_RL_5.pth.tar')
#policy_net.load_state_dict(checkpoint['model_state_dict'])
#
target_net = Q_Network()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
target_net_version = 0
if use_cuda: policy_net, target_net = policy_net.cuda(), target_net.cuda()

optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=1e-6, weight_decay=0.9)
############################################################################################

def partition_video(v_len, v_group, mode): # return is frame index
    frame_indexs = []
    if mode == 'group_random':  # 每个group里随机取一帧
        group_len = v_len // v_group
        for i in range(v_group):
            g_s, g_e = i * group_len, (i + 1) * group_len
            if i == v_group - 1: g_e = v_len
            f_index = random.choice(range(v_len)[g_s:g_e])
            frame_indexs.append(f_index)

    elif mode == 'group_mid':   # 每个group中间取一帧
        pass
    return frame_indexs


def create_random_state(v_a, v_b, label):
    len_a, len_b = len(video_frames_map[v_a]) - 2 * ytf_r, len(video_frames_map[v_b]) - 2 * ytf_r
    a_frame_indexs, b_frame_indexs = partition_video(len_a, Video_Group, mode='group_random'), partition_video(len_b, Video_Group, mode='group_random')
    for i in range(len(b_frame_indexs)): b_frame_indexs[i] += len_a
    h_list = (a_frame_indexs + b_frame_indexs)
    return State(h_list=h_list, v_a=v_a, v_b=v_b, label=label, split_index=len_a, remain_a=Video_Group, remain_b=Video_Group)


def get_droping_frame_path(state, action, r=3):
    if action < state.split_index:
        v_name, remain = state.v_a, state.remain_a
    else:
        v_name, remain = state.v_b, state.remain_b
        action -= state.split_index
    frames = video_frames_map[v_name]
    return v_name + '/' + frames[action + r]


#############  new
def create_video_pair_list():
    # temp implement
    # use global variable train_lines
    v_pair_list = []
    for l in train_lines:
        l = l.strip().split(', ')
        label = 1 if int(l[4]) == 1 else -1
        v_pair_list.append('{},{},{}'.format(l[2], l[3], label))
    return v_pair_list

class Lines_DB(torch.utils.data.Dataset):

    def __init__(self, lines):
        self.lines = lines

    def __getitem__(self, index):
        return self.lines[index]

    def __len__(self):
        return len(self.lines)



def count_reward(pa_feat_cur, pb_feat_cur, pa_feat_next, pb_feat_next, label):
    # count cos
    cos_next = 1-spatial.distance.cosine(pa_feat_next, pb_feat_next)
    cos_cur =  1-spatial.distance.cosine(pa_feat_cur, pb_feat_cur)
    return label * (cos_next - cos_cur)


def count_Qi_init(state):
    a_frames, b_frames = video_frames_map[state.v_a], video_frames_map[state.v_b]
    a_feats, b_feats = [], []
    frame_index_map = {}
    for h_i in state.h_list:
        if h_i < state.split_index:
            f_name = state.v_a + '/' + a_frames[h_i + ytf_r]
            frame_index_map[f_name] = len(a_feats)
            a_feats.append(frame_h_feat_map[f_name])
        else:
            f_name = state.v_b + '/' + b_frames[h_i - state.split_index + ytf_r]
            frame_index_map[f_name] = len(b_feats)
            b_feats.append(frame_h_feat_map[f_name])
    return a_frames, b_frames, a_feats, b_feats, frame_index_map

def get_count_Qi_input(state, action, a_feats, b_feats, frame_index_map):
    # init
    droping_frame_path = get_droping_frame_path(state, action)
    droping_h_feat = frame_h_feat_map[droping_frame_path]

    # handle image
    img = Image.open(os.path.join(YTF_Root_Dir, droping_frame_path)).convert("RGB")
    trans = transforms.ToTensor()
    img = trans(img).unsqueeze(dim=0)

    # filter action
    filter_index = frame_index_map[droping_frame_path]
    if action < state.split_index:
        a_feats[filter_index], a_feats[-1] = a_feats[-1], a_feats[filter_index]; a_feats.pop()
    else:
        b_feats[filter_index], b_feats[-1] = b_feats[-1], b_feats[filter_index]; b_feats.pop()

    # count mean var
    pa_feat, pb_feat = np.mean(a_feats, axis=0), np.mean(b_feats, axis=0)
    var_a_feat, var_b_feat = np.var(a_feats, axis=0), np.var(b_feats, axis=0)

    # recovery filter action
    if action < state.split_index:
        a_feats.append(droping_h_feat); a_feats[filter_index], a_feats[-1] = a_feats[-1], a_feats[filter_index];
    else:
        b_feats.append(droping_h_feat); b_feats[filter_index], b_feats[-1] = b_feats[-1], b_feats[filter_index]

    pa_feat, pb_feat, var_a_feat, var_b_feat, droping_h_feat = torch.from_numpy(pa_feat).unsqueeze(dim=0), torch.from_numpy(pb_feat).unsqueeze(dim=0), torch.from_numpy(var_a_feat).unsqueeze(dim=0), torch.from_numpy(var_b_feat).unsqueeze(dim=0), torch.from_numpy(droping_h_feat).unsqueeze(dim=0)
    return img, pa_feat, pb_feat, var_a_feat, var_b_feat, droping_h_feat



def select_action_count_reward_and_Qi(state, Q_fun): # 输入的state里，a和b不能帧数都是1
    # init param
    a_frames, b_frames, a_feats, b_feats, frame_index_map = count_Qi_init(state)
    pa_feat_cur, pb_feat_cur = np.mean(a_feats, axis=0), np.mean(b_feats, axis=0)

    # redine action space
    h_list = state.h_list
    if state.remain_a == 1: h_list = [h for h in h_list if not h < state.split_index]
    if state.remain_b == 1: h_list = [h for h in h_list if  h < state.split_index]

    # each action count Q and r
    with torch.no_grad():
        Q_i_arr, reward_arr = [], []
        for i in range(0, len(h_list), act_batch):
            start2 = time.time()
            imgs, pa_feats, pb_feats, var_a_feats, var_b_feats, droping_h_feats = [], [], [], [], [], []
            for temp_action in h_list[i:i+act_batch]:
                img, pa_feat, pb_feat, var_a_feat, var_b_feat, droping_h_feat = get_count_Qi_input(state, temp_action, a_feats, b_feats, frame_index_map)
                imgs.append(img); pa_feats.append(pa_feat); pb_feats.append(pb_feat); var_a_feats.append(var_a_feat); var_b_feats.append(var_b_feat); droping_h_feats.append(droping_h_feat)
                reward_arr.append(count_reward(pa_feat_cur, pb_feat_cur, pa_feat, pb_feat, state.label))
            # wrap batch
            imgs = torch.cat(imgs, 0); pa_feats = torch.cat(pa_feats, 0); pb_feats = torch.cat(pb_feats, 0); var_a_feats = torch.cat(var_a_feats, 0); var_b_feats = torch.cat(var_b_feats, 0); droping_h_feats = torch.cat(droping_h_feats, 0);
            if use_cuda: imgs, pa_feats, pb_feats, var_a_feats, var_b_feats, droping_h_feats = Variable(imgs.cuda()), Variable(pa_feats.cuda()), Variable(pb_feats.cuda()), Variable(var_a_feats.cuda()), Variable(var_b_feats.cuda()), Variable(droping_h_feats.cuda())
            else: imgs, pa_feats, pb_feats, var_a_feats, var_b_feats, droping_h_feats = Variable(imgs), Variable(pa_feats), Variable(pb_feats), Variable(var_a_feats), Variable(var_b_feats), Variable(droping_h_feats)
            Q_i_arr.append(Q_fun(imgs, pa_feats, pb_feats, var_a_feats, var_b_feats, droping_h_feats))
            #print('each action use {} sec'.format(time.time() - start2))
        Q_i_arr = torch.cat(Q_i_arr, 0)
        Q_max_index = torch.max(Q_i_arr, 0)[1].item()

    # select action
    steps_done = pair_steps_map['{},{},{}'.format(state.v_a, state.v_b, state.label)]
    #print('steps_done:{}'.format(steps_done))
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:  # select action based on policy
        action, reward, Q_i = h_list[Q_max_index], reward_arr[Q_max_index], Q_i_arr[Q_max_index].item()
    else:                       # select action random
        Q_random_index = random.choice(range(len(h_list)))
        action, reward, Q_i = h_list[Q_random_index], reward_arr[Q_random_index], Q_i_arr[Q_random_index].item()
    # define next_state_end
    if max(reward_arr) < 0: next_state_end = True
    else:
        next_state_end = False
        remain_a, remain_b = state.remain_a, state.remain_b
        if action < state.split_index: remain_a -= 1
        else: remain_b -= 1
        if remain_a == 1 and remain_b == 1:  next_state_end = True

    return action, reward, Q_i, next_state_end




def thread_run_common(state, model):
    if model == 'policy':
        Q_fun = Q_Network()
        if use_cuda: Q_fun = Q_fun.cuda()
        Q_fun.load_state_dict(policy_net.state_dict())
        Q_fun.eval()


    elif model == 'target':
        Q_fun = Q_Network()
        if use_cuda: Q_fun = Q_fun.cuda()
        Q_fun.load_state_dict(target_net.state_dict())
        Q_fun.eval()

    #start1 = time.time()
    action, reward, Q_i, next_state_end = select_action_count_reward_and_Qi(state, Q_fun)
    reward = reward * 100.0  # 让reward有明显的差异
    #print('select_action_count_reward_and_Qi use {} sec'.format(time.time() - start1))
    return Transition(state=copy.copy(state), action=action, next_state_end=next_state_end, reward=reward, Q_i=Q_i, cache_next_Qi=[])


def mul_select_action_count_reward(batch_pair_input_state):
    transitions = []
    for state in batch_pair_input_state:
        transitions.append(thread_run_common(state, 'policy'))
    return transitions
    # all_task = []
    #     # for state in batch_pair_input_state:
    #     #     all_task.append(executor.submit(thread_run_common, state, 'policy'))
    #     #
    #     # transitions = []
    #     # for future in as_completed(all_task):
    #     #     transitions.append(future.result())
    #     # return transitions

def get_next_state(state, action):
    h_list = [h_i for h_i in state.h_list if h_i != action]
    remain_a, remain_b = state.remain_a, state.remain_b
    if action < state.split_index:  remain_a -= 1
    else: remain_b -= 1
    return State(h_list=h_list, v_a=state.v_a, v_b=state.v_b, label=state.label, split_index=state.split_index, remain_a=remain_a, remain_b=remain_b)


def mul_count_expected_state_values(samples):
    # step 1 only cache_next_Qi is None or cache_next_Qi[0] 不是最新版本的才要重新计算
    # all_task = []
    # for transition in samples:
    #     if not transition.next_state_end:
    #         if transition.cache_next_Qi is None or transition.cache_next_Qi[0] != target_net_version:
    #             state = get_next_state(transition.state, transition.action)
    #             all_task.append(executor.submit(thread_run_common, state, 'target'))
    #
    # next_transitions, next_index = [], 0
    # for future in as_completed(all_task):
    #     next_transitions.append(future.result())

    next_transitions, next_index = [], 0
    for transition in samples:
        if not transition.next_state_end:
            if len(transition.cache_next_Qi) == 0 or transition.cache_next_Qi[0] != target_net_version:
                state = get_next_state(transition.state, transition.action)
                next_transitions.append(thread_run_common(state, 'target'))

    # step 2 count expected_state_action_values (note need update cache)
    bs = len(samples)
    expected_state_action_values = np.zeros((bs,), dtype=np.float32)
    for i, transition in enumerate(samples):
        if not transition.next_state_end:
            if len(transition.cache_next_Qi) == 0 or transition.cache_next_Qi[0] != target_net_version:
                if len(transition.cache_next_Qi) == 0: transition.cache_next_Qi.append(None); transition.cache_next_Qi.append(None);
                transition.cache_next_Qi[0], transition.cache_next_Qi[1] = target_net_version, next_transitions[next_index].Q_i; next_index += 1;
            next_state_value = transition.cache_next_Qi[1]
            expected_state_action_values[i] = (next_state_value * GAMMA) + transition.reward
        else:
            expected_state_action_values[i] = transition.reward
    return torch.from_numpy(expected_state_action_values)   # pytorch shape (batch, )


def optimize_model(samples, expected_state_action_values):
    imgs, pa_feats, pb_feats, var_a_feats, var_b_feats, droping_h_feats = [], [], [], [], [], []
    for transition in samples:
        state, action = transition.state, transition.action
        a_frames, b_frames, a_feats, b_feats, frame_index_map = count_Qi_init(state)
        img, pa_feat, pb_feat, var_a_feat, var_b_feat, droping_h_feat = get_count_Qi_input(state, action, a_feats, b_feats, frame_index_map)
        imgs.append(img); pa_feats.append(pa_feat); pb_feats.append(pb_feat); var_a_feats.append(var_a_feat); var_b_feats.append(var_b_feat); droping_h_feats.append(droping_h_feat);
    imgs = torch.cat(imgs, 0); pa_feats = torch.cat(pa_feats, 0); pb_feats = torch.cat(pb_feats, 0); var_a_feats = torch.cat(var_a_feats, 0); var_b_feats = torch.cat(var_b_feats, 0); droping_h_feats = torch.cat(droping_h_feats, 0);
    # wrap to variable
    if use_cuda: expected_state_action_values, imgs, pa_feats, pb_feats, var_a_feats, var_b_feats, droping_h_feats = Variable(expected_state_action_values.cuda()), Variable(imgs.cuda()), Variable(pa_feats.cuda()), Variable(pb_feats.cuda()), Variable(var_a_feats.cuda()), Variable(var_b_feats.cuda()), Variable(droping_h_feats.cuda())
    else: expected_state_action_values, imgs, pa_feats, pb_feats, var_a_feats, var_b_feats, droping_h_feats = Variable(expected_state_action_values), Variable(imgs), Variable(pa_feats), Variable(pb_feats), Variable(var_a_feats), Variable(var_b_feats), Variable(droping_h_feats)

    # count loss
    state_action_values = policy_net(imgs, pa_feats, pb_feats, var_a_feats, var_b_feats, droping_h_feats)
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-0.5, 0.5)
    optimizer.step()
    return loss


class Manage_Memory(object):

    def __init__(self, v_pair_list):
        pair_memory_map = {}
        for pair in v_pair_list:
            pair_memory_map[pair] = ReplayMemory(capacity=memory_capacity)
        self.pair_memory_map = pair_memory_map

    def record(self, batch_v_pair, transitions):
        pair_memory_map = self.pair_memory_map
        for pair, transition in zip(batch_v_pair, transitions):
            pair_memory_map[pair].push(transition)

    def get_batch_pair_input_state(self, batch_v_pair):
        pair_memory_map = self.pair_memory_map
        batch_pair_input_state = []
        for pair in batch_v_pair:
            transition = pair_memory_map[pair].get_Lately_transition()
            if transition is None or transition.next_state_end == True:
                pair = pair.split(',')  # 0 1 2 is v_a v_b label
                state = create_random_state(pair[0], pair[1], int(pair[2]))
            else:
                state = get_next_state(transition.state, transition.action)
            batch_pair_input_state.append(state)
        return batch_pair_input_state

    def get_optimize_model_samples(self, batch_v_pair):
        pair_memory_map = self.pair_memory_map
        if len(pair_memory_map[batch_v_pair[0]].memory) < Memory_Batch: return None
        samples = []
        for pair in batch_v_pair:
            samples += pair_memory_map[pair].sample(Memory_Batch)
        random.shuffle(samples)
        return samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser("PyTorch Face Recognizer")
    args = parser.parse_args()

    v_pair_list = create_video_pair_list()
    db = Lines_DB(v_pair_list)
    train_loader = torch.utils.data.DataLoader(db, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=False)

    #global pair_steps_map and global manage memory
    pair_steps_map = ADRL_Util.get_pair_steps_map(v_pair_list)
    manage_memory = Manage_Memory(v_pair_list)

    num_episodes = 50000
    for i_episode in range(num_episodes):
        episode_losses = utils.AverageMeter()
        for step in range(Video_Group * 2 - 2):
            epoch_losses = utils.AverageMeter()
            for batch_idx, batch_v_pair in enumerate(train_loader):
                batch_pair_input_state = manage_memory.get_batch_pair_input_state(batch_v_pair)
                transitions = mul_select_action_count_reward(batch_pair_input_state)
                manage_memory.record(batch_v_pair, transitions)
                samples = manage_memory.get_optimize_model_samples(batch_v_pair)  # sample is list contain Transition
                if samples is None: continue
                expected_state_action_values = mul_count_expected_state_values(samples)  # np shape is （batch,）
                loss = optimize_model(samples, expected_state_action_values)
                # log
                episode_losses.update(loss.item()); epoch_losses.update(loss.item());
                log_str = '[i_episode:{} step:{} {}/{}] loss:{:.4f} episode_Loss: {loss1.avg:.4f} epoch_Loss: {loss2.avg:.4f}'.format(
                    i_episode, step, batch_idx, len(train_loader), loss.item(), loss1=episode_losses, loss2=epoch_losses)
                utils.printoneline(utils.dt(), log_str)
            print('')
            pair_steps_map = ADRL_Util.update_pair_steps_map(pair_steps_map)
        save_model(args, policy_net, i_episode)

        # Update the target network
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            target_net_version += 1
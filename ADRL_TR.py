import torch
import torch.nn as nn
import argparse, os, random, cv2, sys, time, math
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from scipy import spatial

sys.path.append('/home/yezilong/my_model/AAA')
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# step 1
def get_frame_feat_map(file='/home/yezilong/reference/caffe-face-my/face_example/ytf_feat_by_center_loss.txt', feat_dim=512):
    frame_feat_map = {}
    with open(file, 'r') as f:
        for l in f:
            l = l.strip().split('\t')
            feat = np.zeros((feat_dim,), dtype=np.float32)
            for i in range(feat_dim):
                try:
                    feat[i] = l[i + 1]
                except ValueError:
                    feat[i] = 1.0
            frame_feat_map[l[0]] = feat
    return frame_feat_map

def get_video_frames_map(root='/home/yezilong/dataset/YouTubeFaces/112X96'):
    video_frames_map = {}
    person_dirs = os.listdir(root)
    for person_dir in person_dirs:
        video_dirs = os.listdir(os.path.join(root, person_dir))
        for video_dir in video_dirs:
            frames = sorted(os.listdir(os.path.join(root, person_dir, video_dir)), key=lambda f: int(f.split('.')[-2]))
            v_name = '{}/{}'.format(person_dir, video_dir)
            video_frames_map[v_name] = frames
    return video_frames_map


def split_train_test(test_split = 9, split_file = '/home/yezilong/my_model/valid/splits_no_header.txt'): # test_split is [0-9]
    with open(split_file, 'r') as f: lines = f.readlines()
    test_start, test_end = test_split * 500, (test_split + 1) * 500
    test_lines = lines[test_start:test_end]
    # lines minus test_split
    train_lines = [l for l in lines if l not in test_lines]
    return train_lines, test_lines

def get_train_person_videos_map(train_lines):
    train_person_videos_map = {}
    for l in train_lines:
        l = l.split(', ')
        for i in [2, 3]:
            v = l[i].split('/')
            if v[0] not in train_person_videos_map: train_person_videos_map[v[0]] = []
            if v[1] not in train_person_videos_map[v[0]]: train_person_videos_map[v[0]].append(v[1])
    return train_person_videos_map


def get_frame_h_feat_map(h_feat_file='ytf_h_feat.txt', h_feat_dim=1024):
    frame_h_feat_map = {}
    with open(h_feat_file, 'r') as f:
        for l in f:
            l = l.strip().split('\t')
            h_feat = np.zeros((h_feat_dim,), dtype=np.float32)
            for i in range(h_feat_dim):
                try:
                    h_feat[i] = l[i + 1]
                except ValueError:
                    h_feat[i] = 1.0
            frame_h_feat_map[l[0]] = h_feat
    return frame_h_feat_map


# count train video nums
def count_train_video_nums(train_person_videos_map):
    print('person num:{}'.format(len(train_person_videos_map)))
    num = 0
    for _, videos in train_person_videos_map.items():
        num += len(videos)
    print('videos num:{}'.format(num))


##############################################################################################################################
# init global variable
#frame_feat_map = get_frame_feat_map()
frame_feat_map = {}
frame_h_feat_map = get_frame_h_feat_map()

video_frames_map = get_video_frames_map()
train_lines, test_lines = split_train_test()
train_person_videos_map = get_train_person_videos_map(train_lines)

##############################################################################################################################

class YTF_train_db(torch.utils.data.Dataset):

    def gen_random_video_triple(self):
        v_triple = []
        person_list = list(train_person_videos_map.keys())
        for person, videos in train_person_videos_map.items():
            if len(videos) < 2: continue
            for video in videos:
                # random select same person another video
                while True:
                    s_v = random.choice(videos)
                    if s_v != video: break

                # random select different person video
                while True:
                    d_person = random.choice(person_list)
                    d_v = random.choice(train_person_videos_map[d_person])
                    if d_person != person: break

                v_triple.append((person + '/' + video, person + '/' + s_v, d_person + '/' + d_v))
        return v_triple

    def segment_frames(self, frames):
        r, segment = self.r, self.segment # r=3 segment=6
        index = []
        i_range = range(len(frames))
        i_range = i_range[r:-r]
        seg_len = math.floor(len(i_range) / segment)
        for i in range(segment):
            if i == segment - 1: cur_range = i_range[i*seg_len:]
            else: cur_range = i_range[i*seg_len:(i+1)*seg_len]
            cur_index = random.choice(cur_range)
            index.append(cur_index)
        return index

    def gen_random_frame_triple(self, v_triple):
        f_triple = []
        for a_v, s_v, d_v in v_triple:
            a_seg_frame_index = self.segment_frames(video_frames_map[a_v])
            s_seg_frame_index = self.segment_frames(video_frames_map[s_v])
            d_seg_frame_index = self.segment_frames(video_frames_map[d_v])

            for a_frame_index in a_seg_frame_index:
                for s_frame_index in s_seg_frame_index:
                    for d_frame_index in d_seg_frame_index:
                        f_triple.append(('%s|%d'%(a_v, a_frame_index), '%s|%d'%(s_v, s_frame_index), '%s|%d'%(d_v, d_frame_index)))
        return f_triple


    def reset(self):
        v_triple = self.gen_random_video_triple()
        self.f_triple = self.gen_random_frame_triple(v_triple)

    def __init__(self, r = 3, segment = 6):
        self.r = r
        self.segment = segment

    def __len__(self):
        return len(self.f_triple)

    def __getitem__(self, index):
        r = self.r
        result = []
        a_f, s_f, d_f = self.f_triple[index]
        # 分别提取2*r+1帧的特征
        for f in [a_f, s_f, d_f]:
            f = f.split('|')
            frames = video_frames_map[f[0]]
            feats = []
            for i in range(int(f[1]) - r, int(f[1]) + r + 1):
                f_name = f[0] + '/' + frames[i]
                feat = frame_feat_map[f_name]
                feats.append(feat)
            result.append(np.stack(feats, axis=0))
        return result[0], result[1], result[2],


# loss
class Triple_Loss(nn.Module):

    def __init__(self, a=0.4):
        super(Triple_Loss, self).__init__()
        self.a = a

    def forward(self, anchor, positive, negative):   # input (batch 512)
        cos_p, cos_n = F.cosine_similarity(anchor, positive), F.cosine_similarity(anchor, negative)
        L = torch.clamp(self.a - cos_p + cos_n, min = 0.0)
        return torch.mean(L, dim=0)

def save_model(args, model, epoch):
    args.epoch = epoch
    args.arch_type = 'rnn'
    utils.save_epoch(args, model)



# test on YTF
def extract_video_lstm_feat(v_name, rnn, r, mode='m_feat'): # mode is m_feat or h_feats
    frames = video_frames_map[v_name]
    i_range = range(len(frames))
    h_feats = []
    for i in i_range[r:-r]:
        # near feats wrap as input
        near_feats = []
        for j in range(i - r, i + r + 1):
            f_name = v_name + '/' + frames[j]
            feat = frame_feat_map[f_name]
            near_feats.append(feat)
        near_feats = np.stack(near_feats, axis=0)
        input = torch.from_numpy(near_feats).unsqueeze(dim=0)
        input = Variable(input.cuda())

        # count its h_feat
        out, (hn, cn) = rnn(input)
        h_feats.append(torch.mean(out, dim=1).squeeze(dim=0))

    if mode == 'm_feat':   # mean pooling
        m_feat = torch.stack(h_feats, dim=0).mean(dim=0)
        return m_feat.cpu().detach().numpy()
    elif mode == 'h_feats':
        return h_feats



def test_lstm_init():
    # mode 'lstm'
    rnn = nn.LSTM(512, 512, 2, batch_first=True, bidirectional=True).cuda()
    check_path = '/home/yezilong/my_model/AAA/result/checkpoint/rnn_13.pth.tar'
    checkpoint = torch.load(check_path)
    rnn.load_state_dict(checkpoint['model_state_dict'])
    rnn.eval()
    r = 3
    lstm_video_feat_map = {}

    with open('/home/yezilong/my_model/valid/splits_no_header.txt', 'r') as f:
        for l in f:
            l = l.strip().split(', ')
            for i in [2, 3]:
                if l[i] in lstm_video_feat_map: continue
                lstm_video_feat_map[l[i]] = extract_video_lstm_feat(l[i], rnn, r)
    return lstm_video_feat_map



def extract_ytf_feat(v_name, mode, **args):
    frames = video_frames_map[v_name]
    if mode == 'avg':
        feat_dim = 512  # hyper param
        a_feat = np.zeros((feat_dim,), dtype=np.float32)
        for f in frames:
            f_name = v_name + '/' + f
            a_feat += frame_feat_map[f_name]
        a_feat /= len(frames)
        return a_feat

    elif mode == 'lstm':
        r, h_feat_dim = 3, 1024  # hyper param
        a_feat = np.zeros((h_feat_dim,), dtype=np.float32)
        for f in frames[r:-r]:
            f_name = v_name + '/' + f
            a_feat += frame_h_feat_map[f_name]
        a_feat /= (len(frames) - 2 * int(r))
        return a_feat


def test_on_YTF(split_file = '/home/yezilong/my_model/valid/splits_no_header.txt'):
    # start = time.time()
    # lstm_video_feat_map = test_lstm_init()
    # end = time.time()
    # print('use {} second lstm extract feat'.format(end-start))

    predicts = []
    with open(split_file, 'r') as f:
        i = 0
        for l in f:
            l = l.strip().split(', ')
            #f1, f2 = extract_ytf_feat(l[2], mode='avg'), extract_ytf_feat(l[3], mode='avg')
            f1, f2 = extract_ytf_feat(l[2], mode='lstm'), extract_ytf_feat(l[3], mode='lstm')
            #f1, f2 = lstm_video_feat_map[l[2]], lstm_video_feat_map[l[3]]
            a_score = 1-spatial.distance.cosine(f1, f2)
            predicts.append('{}\t{}\t{}\t{}\n'.format(l[2],l[3],a_score,l[4]))
            sys.stdout.write('\r%d/5000' % (i)), sys.stdout.flush()
            i += 1

    accuracy = []
    for test_split in range(10):
        thresholds = np.arange(-1.0, 1.0, 0.005)
        test_start, test_end = test_split * 500, (test_split + 1) * 500
        test_lines = predicts[test_start:test_end]
        train_lines = [l for l in predicts if l not in test_lines]

        best_thresh = utils.find_best_threshold(thresholds, train_lines)
        acc = utils.eval_acc(best_thresh, test_lines)
        print('YTF test_split={} accuracy={:.4f} thresh={:.4f}'.format(test_split, acc, best_thresh))
        accuracy.append(acc)
    print('mean ACC={:.4f}'.format(np.mean(accuracy)))


# save ytf h_feat
def save_ytf_h_feat(h_feat_file='ytf_h_feat1.txt'):
    rnn = nn.LSTM(512, 512, 2, batch_first=True, bidirectional=True).cuda()
    check_path = '/home/yezilong/my_model/AAA/result/checkpoint/rnn_13.pth.tar'
    checkpoint = torch.load(check_path)
    rnn.load_state_dict(checkpoint['model_state_dict'])
    rnn.eval()
    r, h_feat_dim = 3, 1024
    lstm_video_feat_map = {}

    h_feat_file = open(h_feat_file, 'w')
    with open('/home/yezilong/my_model/valid/splits_no_header.txt', 'r') as f:
        for l in f:
            l = l.strip().split(', ')
            for i in [2, 3]:
                if l[i] in lstm_video_feat_map: continue
                lstm_video_feat_map[l[i]] = []  # only avoid count twice

                # extract h feats
                h_feats = extract_video_lstm_feat(l[i], rnn, r, mode='h_feats')
                # start save in file
                frames = video_frames_map[l[i]]
                i_range = range(len(frames))
                for f_i in i_range[r:-r]:
                    h_feat_file.write(l[i] + '/' + frames[f_i])
                    for j in range(h_feat_dim):
                        h_feat_file.write('\t%f'%(h_feats[f_i-r][j]))
                    h_feat_file.write('\n')
    h_feat_file.close()


if __name__ == '__main__':
    test_on_YTF()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser("PyTorch Face Recognizer")
#     parser.add_argument('--use_cuda', default=True)
#     args = parser.parse_args()
#
#     # dataset
#     db = YTF_train_db()
#     # model
#     rnn = nn.LSTM(512, 512, 2, batch_first=True, bidirectional=True).cuda()
#
#     # resume
#     # check_path = '/home/yezilong/my_model/AAA/result/checkpoint/rnn_7.pth.tar'
#     # checkpoint = torch.load(check_path)
#     # rnn.load_state_dict(checkpoint['model_state_dict'])
#
#     # loss
#     t_loss = Triple_Loss().cuda()
#     # lr
#     optim = torch.optim.SGD(rnn.parameters(), lr=0.0001, momentum=0.4, weight_decay=0)
#
#     # train
#     start_epoch = 8
#     for epoch in range(start_epoch, 22):
#         db.reset()
#         train_loader = torch.utils.data.DataLoader(db, batch_size=128, shuffle=True, num_workers=6, pin_memory=True)
#
#         losses = utils.AverageMeter()
#         rnn.train()
#         for batch_idx, (anchor, positive, negative) in enumerate(train_loader):  # shape (batch, 2*r+1, 512)
#             outs = []
#             for input in [anchor, positive, negative]:
#                 input = Variable(input.cuda())
#                 out, (hn, cn) = rnn(input)
#                 outs.append(torch.mean(out, dim=1))
#             loss = t_loss(outs[0], outs[1], outs[2])
#
#             optim.zero_grad()
#             loss.backward()
#             optim.step()
#
#             # log
#             losses.update(loss.item())
#             log_str = '[epoch:{} {}/{}] loss:{:.4f} Loss: {loss1.avg:.4f} lr {lr:.6f}'.format(
#                         epoch, batch_idx, len(train_loader),
#                         loss.item(), loss1=losses, lr=optim.param_groups[0]['lr'])
#             utils.printoneline(utils.dt(),log_str)
#             if np.isnan(float(loss.item())):
#                 raise ValueError('loss is nan while training')
#
#         save_model(args, rnn, epoch)
#         print('')


































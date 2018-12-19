import numpy as np
import os
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


def get_pair_steps_map(v_pair_list):
    pair_steps_map = {}
    for pair in v_pair_list:
        pair_steps_map[pair] = 0
    return pair_steps_map

def update_pair_steps_map(pair_steps_map):
    for key, value in pair_steps_map.items():
        pair_steps_map[key] = value + 1
    return pair_steps_map

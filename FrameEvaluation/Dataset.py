import torch.utils.data as data


class ADRLDataset(data.Dataset):
    def __getitem__(self, index):
        data=self.ds[index]
        video_names=[data[0],data[1]]
        label=data[2]
        return video_names,label

    def __len__(self):
        return self.ds_len

    def __init__(self,data_list,video_root_dir=''):
        self.ds=[]
        for data in data_list:
            splits=data.strip('\n').split(',')
            if len(splits)!=5:
                continue
            video_a_path=video_root_dir+splits[2].strip()
            video_b_path = video_root_dir+splits[3].strip()
            label=float(splits[4])
            self.ds.append((video_a_path,video_b_path,label))
        self.ds_len=len(self.ds)
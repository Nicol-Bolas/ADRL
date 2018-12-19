import mytool.fileprocess as fp

data_path='E:/work/python/pytorch/ytf_feat_by_center_loss.txt'
new_data_path='E:/work/python/pytorch/ytf_feat_part.txt'

datas=fp.ReadAll(data_path,True)
img_dir=['Paul_Kariya','Talisa_Soto','Manuel_Pellegrini']

keep=[]
print('loaded')
for data in datas:
    if img_dir is not None:
        for dir in img_dir:
            if data.find(dir)>-1:
                keep.append(data)

fp.writeList2file(new_data_path,keep)
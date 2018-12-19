import mytool.fileprocess as fp
import os

data_path='E:/work/python/pytorch/ADRL_data/ytf_h_feat.txt'
person_name_file='E:/work/python/pytorch/ADRL_data/ytf_video_name.txt'
# new_data_dir='E:/work/python/pytorch/ADRL_data/ytf_frame/'

frame_root_dir='E:/work/python/pytorch/ADRL_data/ytf_frame/'

datas=fp.ReadAll(data_path,True)
# person_list=fp.ReadAll(person_name_file,True)
# person_list=[]
# for parent_dir in os.listdir(frame_root_dir):
#     for child_dir in os.listdir(frame_root_dir+parent_dir+'/'):
#         person_list.append(parent_dir+'/'+child_dir)

person_list=['Zulfiqar_Ahmed/1','Zulfiqar_Ahmed/2','Andy_Dick/2']
# print(person_list)
# keep=[]
# person_list=[]
print('loaded')
# for data in datas:
#     frame_path=data.split('\t')[0]
#     splits=frame_path.split('/')
#     person_name=splits[0]+'/'+splits[1]
#
#     if person_name not in person_list:
#         person_list.append(person_name)
#
# fp.writeList2file(person_name_file,person_list)
# print('finish')

fea_list={}

for i in range(len(person_list)):
    fea_list[person_list[i]]=[]

i=0
for data in datas:
    # if i>20:
    #     break
    frame_path = data.split('\t')[0]
    splits=frame_path.split('/')
    person_name=splits[0]+'/'+splits[1]

    if person_name in fea_list.keys():
        fea_list[person_name].append(data)
        i+=1

for k,v in fea_list.items():
    if len(v)>0:
        # splits=k.split('/')
        # name=splits[0]+'-'+splits[1]
        # print(k)
        single_data_file=frame_root_dir+k+'/feat.txt'
        fp.writeList2file(single_data_file,v)

print('finish')
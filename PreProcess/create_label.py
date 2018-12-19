import os
import mytool.fileprocess as fp

video_dir='E:/work/python/pytorch/ADRL_data/ytf_frame/'
label_file='E:/work/python/pytorch/ADRL_data/new_label.txt'

name_list=os.listdir(video_dir)
specific_list=[]

for name in name_list:
    specific_dir=video_dir+name+'/'
    specifics=os.listdir(specific_dir)
    for specific in specifics:
        specific_list.append((name,specific))

label_list=[]
index=1
for i in range(len(specific_list)):
    current=specific_list[i]
    for j in range(i+1,len(specific_list)):
        compare=specific_list[j]
        if current[0]==compare[0]:
            label=1
        else:
            label=-1
        label_list.append('1, '+str(index)+', '+current[0]+'/'+current[1]+', '+compare[0]+'/'+compare[1]+', '+str(label))
        index+=1

fp.writeList2file(label_file,label_list)
print('finish')


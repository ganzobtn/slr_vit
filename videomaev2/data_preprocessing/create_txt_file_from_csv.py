import os
import json
import pickle as pkl
import csv
import pandas as pd



wlasl_videos_path ='/projects/data/slr/autsl/chalearn_processed_full/color/'
kinetics_path = '/projects/data/autsl/cfgs/finetune/videoswin/autsl'
os.makedirs(kinetics_path,exist_ok=True)
#wlasl_videos_write_path = '/projects/data/wlasl_2000/WLASL2000_head_hands_merged/WLASL2000/'




classes  = os.listdir(os.path.join(wlasl_videos_path,'train'))
classes.sort()
print('clas:',classes)
# with open('../misc/label_map_wlasl2000.txt', 'w') as f:
#     for line in classes[:-1]:
#         f.write(f"{line}\n")
#     f.write(f"{classes[-1]}")


#for i in ['val','test','train']:
for i in ['train']:

    with open(os.path.join(kinetics_path,i+'.txt'),'w',newline ='') as file_write:

        for clas in os.listdir(os.path.join(wlasl_videos_path,i)):
            for video in os.listdir(os.path.join(wlasl_videos_path,i,clas)):
                

                path = os.path.join(i, clas,video)
                #path = os.path.join(wlasl_videos_write_path,video)
                label = classes.index(clas)

                print(path,label)
                #if os.path.exists(path):
                file_write.write(path+' '+str(label)+'\n')
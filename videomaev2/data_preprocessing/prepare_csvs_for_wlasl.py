import os
import json
import pickle as pkl
import csv
import pandas as pd



# wlasl_videos_path ='/l/users/ganzorig.batnasan/data/wlasl_2000/wlasl_2000_sorted/'
# kinetics_path = '../datas/cscc/finetune/revised/WLASL2000_head_hands_merged/'
# os.makedirs(kinetics_path,exist_ok=True)
# wlasl_videos_write_path = '/tmp/data/wlasl_2000/WLASL2000_head_hands_merged/WLASL2000/'



wlasl_videos_path ='/projects/data/wlasl_2000/wlasl_2000_sorted/'
kinetics_path = '../datas/dgx/finetune/revised/WLASL2000_head_hands_merged/'
os.makedirs(kinetics_path,exist_ok=True)
wlasl_videos_write_path = '/projects/data/wlasl_2000/WLASL2000_head_hands_merged/WLASL2000/'



for i in ['dev','test','train']:
    for clas in os.listdir(os.path.join(wlasl_videos_path,i)):
        if ' ' in clas:
            os.rename(os.path.join(wlasl_videos_path,i,clas),os.path.join(wlasl_videos_path,i,clas.replace(' ','_')))

classes  = os.listdir(os.path.join(wlasl_videos_path,'train'))
classes.sort()

# with open('../misc/label_map_wlasl2000.txt', 'w') as f:
#     for line in classes[:-1]:
#         f.write(f"{line}\n")
#     f.write(f"{classes[-1]}")


for i in ['dev','test','train']:

    with open(os.path.join(kinetics_path,i+'.csv'),'w',newline ='') as file_write:
        writer = csv.writer(file_write)

        for clas in os.listdir(os.path.join(wlasl_videos_path,i)):
            for video in os.listdir(os.path.join(wlasl_videos_path,i,clas)):
                

                #path = os.path.join(wlasl_videos_path,i, clas,video)
                path = os.path.join(wlasl_videos_write_path,video)
                label = classes.index(clas)

                #print(path,label)
                if os.path.exists(path):
                    writer.writerow([path+' '+str(label)])
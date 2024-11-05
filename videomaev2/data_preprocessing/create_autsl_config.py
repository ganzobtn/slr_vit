import os
import json
import pickle as pkl
import csv
import pandas as pd



wlasl_videos_path ='/tmp/data/autsl/chalearn_processed_full/depth/'
kinetics_path = '../datas/cscc/finetune/revised/autsl/'
os.makedirs(kinetics_path,exist_ok=True)


for i in ['val','test','train']:
    for clas in os.listdir(os.path.join(wlasl_videos_path,i)):
        if ' ' in clas:
            os.rename(os.path.join(wlasl_videos_path,i,clas),os.path.join(wlasl_videos_path,i,clas.replace(' ','_')))

classes  = os.listdir(os.path.join(wlasl_videos_path,'train'))
classes.sort()

# with open('../misc/label_map_autsl.txt', 'w') as f:
#     for line in classes[:-1]:
#         f.write(f"{line}\n")
#     f.write(f"{classes[-1]}")


for i in ['val','test','train']:

    with open(os.path.join(kinetics_path,i+'.csv'),'w+',newline ='') as file_write:
        writer = csv.writer(file_write)

        for clas in os.listdir(os.path.join(wlasl_videos_path,i)):
            for video in os.listdir(os.path.join(wlasl_videos_path,i,clas)):
                

                path = os.path.join(wlasl_videos_path,i, clas,video)
                label = classes.index(clas)

                #print(path,label)
                if os.path.exists(path):
                    writer.writerow([path+' '+str(label)])
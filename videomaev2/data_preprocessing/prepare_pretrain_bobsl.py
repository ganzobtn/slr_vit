import csv
import os
import pandas as pd


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--my_path', type=str, default='/l/users/ganzorig.batnasan/data/kinetics-dataset/k400/annotations')
parser.add_argument('--kinetics_path',type=str,default='../datas/cscc/pretrain/bobsl/')
parser.add_argument('--data_path',type=str,default='/tmp/data/bobsl/videos')
parser.add_argument('--print_paths',action = 'store_true')
opt = parser.parse_args()



my_path = opt.my_path
kinetics_path = opt.kinetics_path
os.makedirs(kinetics_path,exist_ok=True)
data_path = opt.data_path
print_paths = opt.print_paths
print('print_path:',print_paths)

#my_path = '/projects/data/kinetics_dataset/k400/'
for i in ['train.csv']:
    
    rows = []
    with open(os.path.join(kinetics_path,i),'w',newline ='') as file_write:
        writer = csv.writer(file_write)
        #data = pd.read_csv(os.path.join(my_path,i),usecols=['label','youtube_id','time_start','time_end'])
        datas = os.listdir(data_path)
        for video in datas:

            #video_path =  #row['youtube_id'] + '_'+ '0'*(6-len(str(row['time_start'])))+str(row['time_start']) + '_'+ '0'*(6-len(str(row['time_end'])))+str(row['time_end'])  + '.mp4'
            video_path = video

            path = os.path.join(data_path,video_path)
            #print(path,label)
            if print_paths:
                print(path)

            if os.path.exists(path):
                if print_paths:
                    print('exitst:',path)

                # writer.writerow([path,0 , -1 , 0])#+"\, 0\, -1\, 0"]) # video_path, 0, -1, 0
                writer.writerow([path+' 0 -1 0'])#+"\, 0\, -1\, 0"]) # video_path, 0, -1, 0
 
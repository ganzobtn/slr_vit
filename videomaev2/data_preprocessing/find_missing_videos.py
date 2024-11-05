import os
import csv
import pandas as pd
path = '/projects/data/kinetics-dataset/'
my_path = '/projects/data/kinetics_dataset/k400/annotations'
kinetics_path = '../datas/dgx/pretrain/k400'
os.makedirs(kinetics_path,exist_ok=True)
data_path = '/projects/data/kinetics_dataset/k400/'
corrupted_file_path = '/projects/data/kinetics_dataset/k400/replacement/replacement_for_corrupted_k400'
for i in ['train.csv','test.csv','val.csv']:

        print('---------'+i+'---------')
        missing_videos = []
        corrupted_videos = [] 
        data = pd.read_csv(os.path.join(my_path,i),usecols=['label','youtube_id','time_start','time_end'])
        print(len(data))
        for index, row in data.iterrows():
            #print(row['label'], row['youtube_id'], type(row['time_start']),type(row['time_end']))

            assert row['time_start']< row['time_end']
            #print(row['youtube_id'] + '_'+ '0'*(6-len(str(row['time_start'])))+str(row['time_start']) + '_'+ '0'*(6-len(str(row['time_end'])))+str(row['time_end'])  + '.mp4')  
            video_path = row['youtube_id'] + '_'+ '0'*(6-len(str(row['time_start'])))+str(row['time_start']) + '_'+ '0'*(6-len(str(row['time_end'])))+str(row['time_end'])  + '.mp4'

            path = data_path+i.split('.')[0]+'/'+video_path #+', 0, -1, 0'
            if not os.path.exists(path):
                #print(path)
                missing_videos.append(path)
            if os.path.exists(os.path.join(corrupted_file_path,video_path)):
                corrupted_videos.append(video_path)
        print('corrupted:',len(corrupted_videos))   
        print(len(missing_videos))
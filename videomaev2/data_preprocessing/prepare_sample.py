import csv
import os
import pandas as pd

my_path = '/projects/data/kinetics_dataset/k400/annotations'
kinetics_path = '/projects/videomaev2/datas/dgx/pretrain/k400/'
os.makedirs(kinetics_path,exist_ok=True)
data_path = '/projects/data/kinetics_dataset/k400/'

#my_path = '/projects/data/kinetics_dataset/k400/'
for i in ['train.csv','test.csv','val.csv']:
#for i in ['test.csv']:
    missing_videos  = []
    rows = []
    with open(os.path.join(kinetics_path,i),'w',newline ='') as file_write:
        writer = csv.writer(file_write)
        data = pd.read_csv(os.path.join(my_path,i),usecols=['label','youtube_id','time_start','time_end'])
        for index, row in data.iterrows():
            #print(row['label'], row['youtube_id'], type(row['time_start']),type(row['time_end']))
            assert row['time_start']< row['time_end']
            #print(row['youtube_id'] + '_'+ '0'*(6-len(str(row['time_start'])))+str(row['time_start']) + '_'+ '0'*(6-len(str(row['time_end'])))+str(row['time_end'])  + '.mp4')  
            video_path = row['youtube_id'] + '_'+ '0'*(6-len(str(row['time_start'])))+str(row['time_start']) + '_'+ '0'*(6-len(str(row['time_end'])))+str(row['time_end'])  + '.mp4'

            path = data_path+i.split('.')[0]+'/'+video_path #+', 0, -1, 0'
            #print(path,label)
            if os.path.exists(path):
                writer.writerow([path+' 0 -1 0'])#+"\, 0\, -1\, 0"]) # video_path, 0, -1, 0
            else: 
                missing_videos.append(path)

    print(len(missing_videos))
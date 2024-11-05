import os
import json
import pickle as pkl
import csv
import pandas as pd


# It did not work. Pleas fix it if necessary. Just removed the WLASL2000_head_hands_merged folder and renamed WLASL2000 to WLASL2000_head_hands_merged 


kinetics_path = '../datas/cscc/finetune/revised/wlasl_2000/'
old_path = '../datas/cscc/finetune/revised/WLASL2000_head_hands_merged/'
for i in ['val','test','train']:
    with open(os.path.join(old_path,i+'.csv')) as file:
        content = file.readlines()
        #df = pd.read_csv(os.path.join(old_path,i+'.csv'))



        with open(os.path.join(kinetics_path,i+'.csv'),'w',newline ='') as file_write:
            writer = csv.writer(file_write)
            #print(len(df),type(df))
            for row in content:
                #rows.append(row)
                path = row.replace('WLASL2000_head_hands_merged','')
                print(path)
                #path.split(' ')[0] 
                    #print(path,label)
                path = row.split(' ')[0].replace('WLASL2000_head_hands_merged','')
                label = row.split(' ')[1]
                writer.writerow([path + ' '+ str(label)])
                #writer.writerow([path])
                #writer.writerow([path+' '+str(label)])

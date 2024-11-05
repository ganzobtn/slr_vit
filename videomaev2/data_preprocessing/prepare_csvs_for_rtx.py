import os
import json
import pickle as pkl
import csv
import pandas as pd
import argparse


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str,default='../datas/dgx/finetune/revised/wlasl_2000', help='Path to CSV file')
    parser.add_argument('--csv_save_path', type=str, default='../datas/rtx/finetune/wlasl_2000/', help='Path to save the modified CSV file')
    args = parser.parse_args()

    csv_path = args.csv_path
    csv_save_path = args.csv_save_path
    # Create the save directory if it doesn't exist
    os.makedirs(csv_save_path, exist_ok=True)
    # Add your main code logic here
    # ...
    for i in ['dev','test','train']:

        csv_file = os.path.join(csv_path, i+'.csv')                  
        # Read the CSV file
        df = pd.read_csv(csv_file, header=None)

        # Change the first element from each line
        # Get the last 2 elements after splitting by '/'

        df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: '/'.join(x.split('/')[-2:]))

        # Save the modified CSV file
        df.to_csv(os.path.join(csv_save_path,i+'.csv'), index=False,header=None)


import os
from tqdm.auto import tqdm

import pandas as pd

import argparse

def data_preprocessing(args):

    origin_datas = ["Str1", "Str2", "Str3"]
    df = pd.read_csv(os.path.join(args.dataset_path, args.dataset), sep='\t', dtype={'Str1': float,'Str2': float,'Str3': float}, header=0)
    print(df.head())
    for origin_data in origin_datas:
        if origin_data == "Str1":
            data_column_index = 4
        elif origin_data == "Str2":
            data_column_index = 5
        elif origin_data == "Str3":
            data_column_index = 6
        data_column = df.columns[data_column_index]

        if args.time_column_index is not None:
            time_column = df.columns[args.time_column_index]
        
        if args.seq_len is not None:
            datas = df[data_column]
            datas_mean = df[data_column].mean()
            datas_min = df[data_column].min()
            datas_max = df[data_column].max()
            for i in tqdm(range(0, df.shape[0]), desc='{} data preprocessing...'.format(origin_data)):
                df.loc[i,data_column] = round((df[data_column].values[i] - datas_min) * 1000) +5 # for eos sos token
            print(datas_max)
            
    df.to_csv(os.path.join(args.dataset_path, 'preprocessed_data.txt'), sep='\t',columns=["Timestamp","Str1","Str2","Str3"], header=True, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default='./G3')
    parser.add_argument('--dataset', type=str, default='01000002.txt')
    parser.add_argument('--time_column_index', type=int, default=0 )
    parser.add_argument('--seq_len', default=30, type=int)                        
 
 
    args = parser.parse_args()
    data_preprocessing(args)
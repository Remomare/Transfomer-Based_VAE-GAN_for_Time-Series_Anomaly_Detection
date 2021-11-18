import os
from tqdm.auto import tqdm

import pandas as pd
import numpy as np


import torch

class  CustomDataset(torch.utils.data.Dataset):
    def __init__(self, args, data_path:str):

        self.data_path = data_path
        self.datas = []

        df = pd.read_csv(os.path.join(data_path), sep='\t', header=0)

        if args.time_column_index is not None:
            time_column = df.columns[args.time_column_index]
        
        if args.target_column_index is not None:
            target_colunm = df.columns[args.target_column_index]

        if args.data_column_index is not None:

            ingredient_column = df.columns[args.data_column_index]
        
        if args.time_column_index is not None:
            timestamp_column = df.columns[args.time_column_index]
        
        else:
            raise ValueError('data_column_index must be specified if data_path is a csv file')
        
        datas = df[ingredient_column]
        timestamps = df[timestamp_column]
        
        targets = df[target_colunm]
        
        tensor_ingredient = torch.tensor([])
        tensor_timestamp = torch.tensor([])

        for idx in tqdm(df.index, desc=f'Loading {self.data_path}'):
            data = {}

            #data['ingredient'] = torch.tensor(df[ingredient_column][idx], dtype=torch.long)

            ingredient = [datas[idx]]
            timestamp_ = [timestamps[idx]]           
            target = [targets[idx]]
            data['ingredient'] = ingredient
            data['target'] = target

            scala_ingredient = torch.tensor(ingredient, dtype=torch.long)
            scala_timestamp = torch.tensor(timestamp_, dtype=torch.long)

            tensor_ingredient = torch.cat((tensor_ingredient, scala_ingredient))
            tensor_timestamp = torch.cat((tensor_timestamp, scala_timestamp))

            if len(tensor_ingredient) == (args.seq_len):
                
                src_input = tensor_ingredient
                tgt_input = torch.cat([torch.tensor(1, dtype=torch.long).unsqueeze(0), tensor_ingredient]).long()
                tgt_output = torch.cat([tensor_ingredient, torch.tensor(2, dtype=torch.long).unsqueeze(0)]).long()
                timestamp = tensor_timestamp
                timestamp_input = torch.cat([torch.tensor(1, dtype=torch.long).unsqueeze(0), tensor_timestamp]).long()
                timestamp_output = torch.cat([tensor_timestamp, torch.tensor(2, dtype=torch.long).unsqueeze(0)]).long()

                for d, d_name in [(src_input, 'src_input'), (tgt_input, 'tgt_input'), (tgt_output, 'tgt_output'), 
                                  (timestamp, 'timestamp'), (timestamp_input, 'timestamp_input'), (timestamp_output, 'timestamp_output')]:
                    pad_len = (args.max_seq_len+1) - len(d)
                    if pad_len > 0:
                        padding  = torch.tensor(0, dtype=torch.long).unsqueeze(0).repeat(pad_len)
                        data[d_name] = torch.cat([d, padding]).long()
                    elif pad_len == 0:
                        data[d_name] = d.long()
                    else:
                        raise ValueError('Sequence length is greater than max_seq_len')
            
                data['length'] = tgt_output.size(0)
                self.datas.append(data)
                tensor_ingredient = torch.tensor([])
                tensor_timestamp = torch.tensor([])


    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        return self.datas[index]
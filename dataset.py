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
        else:
            raise ValueError('data_column_index must be specified if data_path is a csv file')
        
        datas = df[ingredient_column]
        targets = df[target_colunm]
        times = df[time_column]

        if args.max_seq_len is None:
            args.max_seq_len = max(len(t.split()) for t in datas)
        
        tensor_ingredient = torch.tensor([])
        tensor_target = torch.tensor([])
        tensor_timestamp = torch.tensor([])

        for idx in tqdm(df.index, desc=f'Loading {self.data_path}'):
            data = {}



            #data['ingredient'] = torch.tensor(df[ingredient_column][idx], dtype=torch.long)

            ingredient = [datas[idx]]
            target = [targets[idx]]
            timestamp = [times[idx]]
            data['ingredient'] = ingredient
            data['target'] = target

            if args.time_column_index is not None:
                data['time'] = timestamp

            """
            if args.min_seq_len is not None and len(ingredient) < args.min_seq_len:
                continue
            elif len(ingredient) >= args.max_seq_len - 1: 
                tensor_ingredient = torch.tensor(ingredient[:args.max_seq_len - 1], dtype=torch.long) # Truncate long catpions
            else:
                tensor_ingredient = torch.tensor(ingredient, dtype=torch.long)
            """
            scala_ingredient = torch.tensor(ingredient, dtype=torch.long)
            scala_target = torch.tensor(target, dtype=torch.long)
            scala_timestamp = torch.tensor(timestamp, dtype=torch.long)

            tensor_ingredient = torch.cat((tensor_ingredient,scala_ingredient))
            tensor_target = torch.cat((tensor_target, scala_target))
            tensor_timestamp = torch.cat((tensor_timestamp, scala_timestamp))

            if len(tensor_ingredient) == (args.seq_len):

                if args.vae_setting==True:
                    src_input = tensor_ingredient
                    target_input = tensor_target
                    tgt_input = torch.cat([torch.tensor(1, dtype=torch.long).unsqueeze(0), tensor_ingredient]).long()
                    tgt_output = torch.cat([tensor_ingredient, torch.tensor(2, dtype=torch.long).unsqueeze(0)]).long()
                    timestamp = tensor_timestamp

                    for d, d_name in [(src_input, 'src_input'), (tgt_input, 'tgt_input'), (tgt_output, 'tgt_output'), (target_input, 'target_input'), (timestamp, 'timestamp')]:
                        pad_len = args.max_seq_len - len(d)
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
                    tensor_target = torch.tensor([])
                    tensor_timestamp = torch.tensor([])

                else:
                    src_input = tensor_ingredient
                    tgt_input = torch.cat([torch.tensor(1, dtype=torch.long).unsqueeze(0), tensor_ingredient]).long()
                    tgt_output = torch.cat([tensor_ingredient, torch.tensor(2, dtype=torch.long).unsqueeze(0)]).long()
                    for d, d_name in [(src_input, 'src_input'),(tgt_input, 'tgt_input'), (tgt_output, 'tgt_output')]:

                        pad_len = (args.max_seq_len+1) - len(d)
                        if pad_len > 0:
                            padding  = torch.tensor(0, dtype=torch.long).unsqueeze(0).repeat(pad_len)
                            data[d_name] = torch.cat([d, padding]).long()
                        elif pad_len == 0:
                            data[d_name] = d.long()

                        else:
                            raise ValueError('Sequence length is greater than max_seq_len')

                    data['length'] = src_input.size(0)
                    self.datas.append(data)
                    tensor_ingredient = torch.tensor([])
                    tensor_target = torch.tensor([])
                    tensor_timestamp = torch.tensor([])

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        return self.datas[index]
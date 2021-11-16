import os
from torch._C import dtype
from tqdm.auto import tqdm

import pandas as pd
import numpy as np


import torch

class  CustomDataset(torch.utils.data.Dataset):
    def __init__(self, args, data_path:str):

        self.data_path = data_path
        self.datas_1 = []
        self.datas_2 = []

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
        
        datas_1 = df[ingredient_column]
        datas_2 = df[ingredient_column+1]
        targets = df[target_colunm]
        timestamps = df[timestamp_column]
        
        if args.max_seq_len is None:
            args.max_seq_len = max(len(t.split()) for t in datas_1)
        
        tensor_ingredient_1 = torch.tensor([])
        tensor_ingredient_2 = torch.tensor([])
        tensor_target = torch.tensor([])
        tensor_timestamp = torch.tensro([])

        
        for idx in tqdm(df.index, desc=f'Loading {self.data_path}'):
            data = {}

            #data['ingredient'] = torch.tensor(df[ingredient_column][idx], dtype=torch.long)

            ingredient_1 = [datas_1[idx]]
            ingredient_2 = [datas_2[idx]]
            target = [targets[idx]]
            timestamp_ = [timestamps[idx]]
            data['ingredient_1'] = ingredient_1
            data['ingredient_2'] = ingredient_2
            data['target'] = target
            data['time'] = timestamp_

            """
            if args.min_seq_len is not None and len(ingredient) < args.min_seq_len:
                continue
            elif len(ingredient) >= args.max_seq_len - 1: 
                tensor_ingredient = torch.tensor(ingredient[:args.max_seq_len - 1], dtype=torch.long) # Truncate long catpions
            else:
                tensor_ingredient = torch.tensor(ingredient, dtype=torch.long)
            """
            scala_ingredient_1 = torch.tensor(ingredient_1, dtype=torch.long)
            scala_ingredient_2 = torch.tensor(ingredient_2, dtype=torch.long)
            scala_target = torch.tensor(target, dtype=torch.long)
            scala_timestamp = torch.tensor(timestamp_, dtype=torch.long)

            tensor_ingredient_1 = torch.cat((tensor_ingredient_1,scala_ingredient_1))
            tensor_ingredient_2 = torch.cat((tensor_ingredient_2,scala_ingredient_2))
            tensor_timestamp = torch.cat((tensor_timestamp, scala_timestamp))

            tensor_target = torch.cat((tensor_target, scala_target))


            if len(tensor_ingredient_1) == (args.seq_len):

                if args.vae_setting==True:
                    src_input_1 = tensor_ingredient_1
                    src_input_2 = tensor_ingredient_2
                    timestamp = tensor_timestamp
                    tgt_input_1 = torch.cat([torch.tensor(1, dtype=torch.long).unsqueeze(0), tensor_ingredient_1]).long()
                    tgt_output_1 = torch.cat([tensor_ingredient_1, torch.tensor(2, dtype=torch.long).unsqueeze(0)]).long()
                    tgt_input_2 = torch.cat([torch.tensor(1, dtype=torch.long).unsqueeze(0), tensor_ingredient_2]).long()
                    tgt_output_2 = torch.cat([tensor_ingredient_2, torch.tensor(2, dtype=torch.long).unsqueeze(0)]).long()

                    for d, d_name in [(src_input_1, 'src_input_1'), (tgt_input_1, 'tgt_input_1'), (tgt_output_1, 'tgt_output_1'), 
                                      (src_input_2, 'src_input_2'), (tgt_input_2, 'tgt_input_2'), (tgt_output_2, 'tgt_output_2'),
                                      (timestamp, 'timestamp')]:
                        pad_len = (args.max_seq_len+1) - len(d)
                        if pad_len > 0:
                            padding  = torch.tensor(0, dtype=torch.long).unsqueeze(0).repeat(pad_len)
                            data[d_name] = torch.cat([d, padding]).long()
                        elif pad_len == 0:
                            data[d_name] = d.long()
                        else:
                            raise ValueError('Sequence length is greater than max_seq_len')
            
                    data['length'] = tgt_output_1.size(0)
                    self.datas.append(data)
                    tensor_ingredient_1 = torch.tensor([])
                    tensor_ingredient_2 = torch.tensor([])
                    tensor_target = torch.tensor([])
                    tensor_timestamp = torch.tensor([])
                """
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
                """

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        return self.datas[index]
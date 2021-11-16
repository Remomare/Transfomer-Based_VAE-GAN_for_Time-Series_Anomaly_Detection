import os
import argparse
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataset import CustomDataset
from model import Transformer
from epoch import test_model
from utils import  set_random_seed, get_tb_experiment_name

def main(args):
    set_random_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    csv_path = os.path.join(args.dataset_path + args.dataset)
    dataset_test = CustomDataset(args, csv_path)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, 
                                                drop_last=False, shuffle=False, num_workers=args.num_workers)

    model = Transformer(args=args, batch_size=args.batch_size, vocab_size=args.vocab_size+4,
                        embed_size=args.embed_size, hidden_size=args.hidden_size, latent_size=args.latent_size,
                        embedding_dropout_ratio=args.embedding_dropout_ratio, num_layers=args.num_layers,
                        topk=args.topk, vae_setting=args.vae_setting,
                        device=device).to(device) 
    
    model.load_state_dict(torch.load(args.model_path))

    if args.use_tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.tensorboard_log_dir, get_tb_experiment_name(args)))
        writer.add_text('model', str(model))
        writer.add_text('args', str(args))
        test_model(args, model, dataloader_test, writer, device)
    else:
        test_model(args, model, dataloader_test, writer=None, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='Transformer_time_series', type=str, 
                        help='Model name') 
    parser.add_argument('--load_from_checkpoint', default=None, type=str,
                        help='Path of existing model')
    parser.add_argument('--save_checkpoint_path', default='./model_checkpoint/', type=str,
                        help='Path to save model checkpoint')
    parser.add_argument('--save_result_path', default='./model_result/', type=str,
                        help='Path to save final model')
    parser.add_argument('--debug_path', default='./model_debug', type=str,
                        help='Path to save debug files')

    parser.add_argument('--model_path', default='./model_result/Transformer_time_series.pt', type=str)

    parser.add_argument('--dataset_path', type=str,
                        default='./G3/',
                        help='Path of dataset foler')
    parser.add_argument('--dataset', type=str,
                        default='preprocessed_data.txt',
                        help='Specific dataset to use')
    parser.add_argument('--time_column_index', type=int, default=0,
                        help='Column index of label in csv file')
    parser.add_argument('--data_column_index', type=int, default=1,
                        help='Column index of text in csv file. Must be given if dataset_path is .csv format')
    parser.add_argument('--target_column_index', default=3, type=int)
    parser.add_argument('--vocab_size', default=48000, type=int,
                        help='Caption vocabulary size; Default is 8000')
    parser.add_argument('--max_seq_len', default=100, type=int,
                        help='maximum sequence length for each sequence')
    parser.add_argument('--seq_len', default=30, type=int)
    parser.add_argument('--min_seq_len', default=10, type=int,
                        help='minumum sequence length for each sequence')
    parser.add_argument('--topk', default=1, type=int,
                        help='topk for decoder')

    parser.add_argument('--embed_size', default=512, type=int,
                        help='Size of embedding vector for model')
    parser.add_argument('--hidden_size', default=256, type=int,
                        help='Size of hidden vector for model')
    parser.add_argument('--latent_size', default=32, type=int,
                        help='Size of latent vector for model')
    parser.add_argument('--embedding_dropout_ratio', default=0.5, type=float,
                        help='Dropout ratio for embedding layer')
    parser.add_argument('--num_layers', default=6, type=int)


    parser.add_argument('--vae_setting', default=False, type= bool)

    parser.add_argument('--epoch', default=30, type=int,
                        help='Epoch size for training')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--lr', default=0.0010, type=float,
                        help='Learning rate for optimizer')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed for system')
    parser.add_argument('--kl_anneal_function', default='logistic', type=str,
                        choices=['logistic', 'linear', 'none'],
                        help='kl_anneal_function')

    parser.add_argument('--save_best_only', default=False, type=bool,
                        help='save best valid accuracy only')
    parser.add_argument('--early_stopping_patience', default=None, type=int,
                        help='patience to stop training')
    parser.add_argument('--log_interval', default=500, type=int,
                        help='Interval for printing batch loss')
    parser.add_argument('--use_tensorboard_logging', default=True, type=bool,
                        help='use tensorboard for logging')
    parser.add_argument('--tensorboard_log_dir', default='runs', type=str)
    parser.add_argument('--save_gradient_flow', default=False, type=bool,
                        help='save gradient flow for debugging')
    parser.add_argument('--show_all_tensor', default=False, type=bool,
                        help='torch.set_printoptions(profile="full") if True')
    parser.add_argument('--set_detect_anomaly', default=False, type=bool,
                        help='torch.autograd.set_detect_anomaly(True) if True')

    args = parser.parse_args()

    if not os.path.exists(args.save_checkpoint_path):
        os.makedirs(args.save_checkpoint_path)
    if not os.path.exists(args.save_result_path):
        os.makedirs(args.save_result_path)
    if not os.path.exists(args.debug_path):
        os.makedirs(args.debug_path)

    if args.show_all_tensor:
        torch.set_printoptions(profile="full")
    if args.set_detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    main(args)
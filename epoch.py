import os
import pandas as pd
from tqdm.auto import tqdm
import torch
import matplotlib.pyplot as plt

from utils import batch_accuracy_test, plot_grad_flow, vae_batch_accuracy

kl_anneal_step = 0
best_acc = 0
best_epoch_idx = -1
early_stopping_cnt = 0

def train_epoch(args, epoch_idx, model, dataloader, optimizer, scheduler, loss_fn, writer, device):

    model = model.train()

    global kl_anneal_step
    epoch_acc = 0
    if args.vae_setting == True:
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'TRAIN EPOCH {epoch_idx}/{args.epoch}')):
            src_input = batch['src_input'].to(device)
            tgt_input = batch['tgt_input'].to(device)
            tgt_output = batch['tgt_output'].to(device)
            time = batch['time'].to(device)
            length = batch['length'].to(device)

            log_prob, mean, log_var, z = model(src_input, tgt_input, length, time)
            NLL_loss, KL_loss, KL_weight = loss_fn(log_prob, tgt_output, length, mean, log_var, kl_anneal_step)
            loss = (NLL_loss + KL_weight * KL_loss) / args.batch_size

            batch_acc = vae_batch_accuracy(log_prob, tgt_output, length)
            epoch_acc += batch_acc.detach()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            kl_anneal_step += 1

        # Logging
            if batch_idx % args.log_interval == 0 or batch_idx == len(dataloader) - 1:
                tqdm.write(f'TRAIN: {batch_idx}/{len(dataloader)} - Loss={loss.item()} \
                        NLL_loss={NLL_loss.item()/args.batch_size} KL_Loss={KL_loss.item()/args.batch_size}')
            if args.save_gradient_flow:
                plot_grad_flow(model.named_parameters())
            if args.use_tensorboard_logging:
                total_idx = batch_idx + (epoch_idx * len(dataloader))
                writer.add_scalar('TRAIN/ELBO', loss.item() / args.batch_size, total_idx)
                writer.add_scalar('TRAIN/NLL_loss', NLL_loss.item() / args.batch_size, total_idx)
                writer.add_scalar('TRAIN/KL_Loss', KL_loss.item() / args.batch_size, total_idx)
                writer.add_scalar('TRAIN/KL_Weight', KL_weight, total_idx)
                writer.add_scalar('TRAIN/Batch_Accuracy', batch_acc.item(), batch_idx+(epoch_idx*len(dataloader)))
    else:
         for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'TRAIN EPOCH {epoch_idx}/{args.epoch}')):
            src_input = batch['src_input'].to(device)
            tgt_input = batch['tgt_input'].to(device)
            tgt_output = batch['tgt_output'].to(device)
            time = batch['time'].to(device)
            length = batch['length'].to(device)
            """
            need check
            """
            z, output, tgt_embedding = model(src_input, tgt_input, length, time)

            loss = loss_fn(output, tgt_embedding)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
  
          # Logging
            if batch_idx % args.log_interval == 0 or batch_idx == len(dataloader) - 1:
                tqdm.write(f'TRAIN: {batch_idx}/{len(dataloader)} - Loss={loss.item()}')
            if args.save_gradient_flow:
                plot_grad_flow(model.named_parameters())
            if args.use_tensorboard_logging:
                total_idx = batch_idx + (epoch_idx * len(dataloader))
                writer.add_scalar('TRAIN/NLL_loss', loss.item() / args.batch_size, total_idx)
                #writer.add_scalar('TRAIN/Batch_Accuracy', batch_acc.item(), batch_idx+(epoch_idx*len(dataloader)))


    epoch_acc /= len(dataloader)
    if args.use_tensorboard_logging:
        writer.add_scalar('TRAIN/Epoch_Accuracy', epoch_acc, epoch_idx)

    if args.save_gradient_flow:
        tqdm.write(f'Logging gradient flow to {args.debug_path}/gradient_flow_epoch_{epoch_idx}.png')
        plt.savefig(os.path.join(args.debug_path, ("gradient_flow_epoch_" + str(epoch_idx) + ".png")), bbox_inches='tight')
        plt.clf()
        
   
def test_model(args, model, dataloader, spm_model, writer, device):

    model = model.eval()
    total_bleu_score = 0
    reference_text = []
    generated_text = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc='TEST Sequence')):
        src_input = batch['src_input'].to(device)
        tgt_input = batch['tgt_input'].to(device)
        tgt_output = batch['tgt_output'].to(device)
        label = batch['label'].to(device)
        length = batch['length'].to(device)
        reference = batch['text']
        batch_bleu_score = 0

        if args.vae_setting == True:
            with torch.no_grad():
                src_embedding = model.get_embedding(src_input)

                src_mask = model.encoder.generate_square_subsequent_mask(src_input.size(1))
                src_pad_mask = model.encoder.generate_padding_mask(src_input, spm_model.pad_id())

                z, mean, log_var = model.encoder(src_embedding, src_mask, src_pad_mask)
                output = model.decoder.decode(z, src_pad_mask)

        if args.vae_setting ==False:
            with torch.no_grad():
                src_embedding = model.get_embedding(src_input)

                src_mask = model.encoder.generate_square_subsequent_mask(src_input.size(1))
        for i in range(0, src_input.size(0)):
            reference_text.append(reference[i])
            generated_text.append(output[i])
            source_distance = generated_text - reference_text

        # Logging
        if batch_idx % args.log_interval == 0 or batch_idx == len(dataloader) - 1:
            tqdm.write(f'TEST: {batch_idx}/{len(dataloader)} - BLEU={source_distance}')
        if args.use_tensorboard_logging:
            writer.add_scalar('TEST/BLEU_Score', source_distance, batch_idx)

    total_bleu_score /= len(dataloader)

    tqdm.write("Completed model test")
    if args.use_tensorboard_logging:
        writer.add_text('TEST/Total_BLEU_Score', str(total_bleu_score))
    
    # Save generated text as csv file
    if args.data_path.endswith('.csv'):
        df_reference_logit = pd.read_csv(os.path.join(args.data_path), header=None)
        df_generated = pd.DataFrame(generated_text)
        df_reference_text = pd.DataFrame(reference_text)

        df = pd.concat([df_reference_logit[0], df_reference_text, df_generated], axis=1)
        df.to_csv(args.output_path, header=None, index=None)
    elif args.data_path.endswith('.txt'):
        with open(args.data_path, 'w') as f:
            for sentence in generated_text:
                f.write(sentence + '\n')
    tqdm.write(f"Saved generated text to {args.output_path}")
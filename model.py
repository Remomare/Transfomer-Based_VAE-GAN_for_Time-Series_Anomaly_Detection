import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, args, batch_size, vocab_size=8004, embed_size=512, hidden_size=256, nhead=8, latent_size=32, 
                embedding_dropout_ratio=0.5, num_layers=6, topk=1,
                device='cuda' if torch.cuda.is_available() else 'cpu'):

        super(Transformer, self).__init__()
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.latent_size = latent_size
        self.embedding_dropout_ratio = embedding_dropout_ratio
        self.num_layers = num_layers
        self.topk = topk

        
        self.args = args
        self.device = device

        self.data_embed = nn.Embedding(vocab_size, embed_size)
        self.embedding_dropout = nn.Dropout(p=self.embedding_dropout_ratio)

        self.encoder = encoderTransformer(args, batch_size, embed_size, nhead, latent_size, num_layers, device)
        self.decoder = decoderTransformer(args, batch_size, embed_size, nhead, latent_size, num_layers, device)

    def forward(self, input_data, target_data, non_pad_length, time):

        input_embedding = self.data_embed(input_data)
        target_embedding = self.data_embed(target_data)

        if self.args.vae_setting == True:
            src_mask = self.encoder.generate_square_subsequent_mask(input_data.size(1))
            src_pad_mask = self.encoder.generate_padding_mask(input_data, 0)
            tgt_mask = self.decoder.generate_square_subsequent_mask(target_data.size(1))
            tgt_pad_mask = self.decoder.generate_padding_mask(target_data, 0)

            if self.embedding_dropout.p > 0:
                target_embedding = self.embedding_dropout(target_embedding)
    
            z, mean, log_var = self.encoder(input_embedding, src_mask=src_mask, src_pad_mask=src_pad_mask)
            log_prob = self.decoder(z, target_embedding, tgt_mask=tgt_mask, mem_mask=src_mask, tgt_pad_mask=tgt_pad_mask, mem_pad_mask=src_pad_mask)
            return log_prob, mean, log_var, z

        else:
            src_mask = self.encoder.generate_square_subsequent_mask(input_data.size(1))
            src_pad_mask = self.encoder.generate_padding_mask(input_data, 0)
            tgt_mask = self.decoder.generate_square_subsequent_mask(target_data.size(1)) 
            tgt_pad_mask = self.decoder.generate_padding_mask(target_data, 0)
            
            z = self.encoder(input_embedding, src_mask, src_pad_mask)
            
            if self.embedding_dropout.p != 0:
                target_embedding = self.embedding_dropout(target_embedding)
            
            output = self.decoder(z, target_embedding, tgt_mask=tgt_mask, mem_mask=src_mask, tgt_pad_mask=tgt_pad_mask, mem_pad_mask=src_pad_mask)
            return output



    def get_embedding(self, input_data):
        return self.text_embed(input_data)



class encoderTransformer(nn.Module):
    def __init__(self, args, batch_size, input_size, nhead, latent_size, num_layers, device):
        super(encoderTransformer, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.nhead = nhead
        self.latent_size= latent_size
        self.num_layers = num_layers
        self.args =args
        self.device = device

        self.src_mask = None

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=input_size*4, batch_first=True, device=device)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.pos_encoder = positionalEncoding(input_size)

        self.linear_layer1 = nn.Linear(input_size, latent_size * 2)
        self.linear_layer2 = nn.Linear(latent_size * 2, latent_size)

        self.activation_function = nn.Sequential(self.linear_layer1, nn.GELU(), self.linear_layer2)

    def forward(self, input_embedding, src_mask, src_pad_mask, condition_embedding=None, has_mask = True):
        
        if self.args.vae_setting == True: #transformer-based-vae
            mean, log_var = self.vae_encode(input_embedding, src_mask, src_pad_mask, condition_embedding)
            z = self.reparameterize(mean, log_var)
            return z, log_var, mean
        else: #transformer
            if has_mask:
                if self.src_mask is None or self.src_mask.size(0) != len(input_embedding):
                    mask = self.generate_square_subsequent_mask(len(input_embedding))
                    self.src_mask = mask
            else:
                self.src_mask = None

            z = self.encoder(input_embedding, src_mask, src_pad_mask)

            return z

    def generate_padding_mask(self, data, pad_idx):

            padding_mask = (data == pad_idx)

            return padding_mask

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(self.device) 
        return mask
            
    def encoder(self, input_embedding, src_mask, src_pad_mask):

        input_embedding = input_embedding * math.sqrt(self.input_size)
        input_embedding = self.pos_encoder(input_embedding)

        encoder_output = self.transformer_encoder(input_embedding, src_mask, src_key_padding_mask=src_pad_mask)

        return encoder_output

    def vae_encode(self, input_embedding, src_mask, src_pad_mask, condition_embedding):
        
        batch_size = input_embedding.size(0)

        input_embedding = self.pos_encoder(input_embedding) 


        if condition_embedding is not None:
            for i in range(input_embedding.size(0)):
                input_embedding[i, :, :] = input_embedding[i, :, :] + condition_embedding[i]

        hidden = self.transformer_encoder(input_embedding, mask=src_mask, src_key_padding_mask=src_pad_mask)
        
        mean = self.activation_function(hidden)
        log_var = self.activation_function(hidden) 

        return mean, log_var

    def reparameterize(self, mean, log_var):

        batch_size = mean.size(0)
        seq_len = mean.size(1)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, seq_len, self.latent_size]).to(self.device) 
        z = eps * std + mean 

        return z



class decoderTransformer(nn.Module):
    def __init__(self, args, batch_size, input_size, nhead,  latent_size, num_layers, device, topk=1):
        super(decoderTransformer, self).__init__()
        self.args = args
        self.batch_size = batch_size
        self.input_size = input_size # same as embed_size
        self.nhead = nhead
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.topk = topk
        self.device = device

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=input_size*4, batch_first=True, device=device)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.pos_encoder = positionalEncoding(input_size)


        self.linear_layer1 = nn.Linear(latent_size, input_size // 8)
        self.linear_layer2 = nn.Linear(input_size // 8, input_size)

        self.activation_function = nn.Sequential(self.linear_layer1, nn.GELU(), self.linear_layer2)

    def forward(self, z, target_embedding, tgt_mask, mem_mask, tgt_pad_mask, mem_pad_mask):
        
        batch_size = target_embedding.size(0)

        target_embedding = self.pos_encoder(target_embedding)
        
        if self.args.vae_setting == True:

            hidden = self.activation_function(z) 

            output = self.transformer_decoder(tgt=target_embedding, memory=hidden, tgt_mask=tgt_mask, memory_mask=mem_mask, tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=mem_pad_mask) 
            logits = self.activation_function(output) 
            log_prob = F.log_softmax(logits, dim=-1) 

            return log_prob
        
        else:

            output = self.transformer_decoder(tgt=target_embedding, memory=z, tgt_mask=tgt_mask, memory_mask=mem_mask, tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=mem_pad_mask) 
            return output

    def vae_decode(self, z, mem_pad_mask):
         
        batch_size = z.size(0)
        seq_len = z.size(1)

        hidden = self.activation_function(z)

        output = torch.ones(batch_size, seq_len).long().to(self.device)

        for i in range(1, seq_len):
            tgt_embedding = self.embed_layer(output[:, :i]) 
            tgt_embedding = self.pos_encoder(tgt_embedding) 

            tgt_mask = self.generate_square_subsequent_mask(length=i)

            decoder_output = self.transformer_decoder(tgt=tgt_embedding, memory=hidden, tgt_mask=tgt_mask, tgt_key_padding_mask=None, memory_key_padding_mask=mem_pad_mask) 
            pred_prob = self.activation_function(decoder_output)
    
            pred_prob = pred_prob[:, -1, :] 
            
            if self.topk == 1:
                output_t = pred_prob.data.topk(self.topk).indices.squeeze() 
            else:
                topk_prob, topk_indices = pred_prob.data.topk(self.topk, dim=-1) 
                sampled = torch.multinomial(topk_prob, num_samples=1).squeeze() 
                output_t = torch.zeros(batch_size).long().to(self.device)
                for j in range(batch_size):
                    output_t[j] = topk_indices[j, sampled[j]]
            output[:, i] = output_t
        
        return output

    def generate_padding_mask(self, data, pad_idx):

            padding_mask = (data == pad_idx)

            return padding_mask

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(self.device) 
        return mask



class positionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(positionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, args, batch_size, vocab_size, embed_size, hidden_size, nhead, latent_size, 
                embedding_dropout_ratio, num_layers, topk, vae_setting,
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
        self.vae_setting = vae_setting
        self.device = device
        
        self.data_embed = nn.Embedding(vocab_size, embed_size)
        self.embedding_dropout = nn.Dropout(p=self.embedding_dropout_ratio)

        self.encoder = encoderTransformer(args, batch_size, embed_size, nhead, latent_size, num_layers, device)
        self.decoder = decoderTransformer(args, batch_size, embed_size, nhead, latent_size, vocab_size, self.data_embed, num_layers, device)

    def forward(self, input_data, target_data, non_pad_length, timestamp):

        input_embedding = self.data_embed(input_data) * math.sqrt(self.embed_size)
        target_embedding = self.data_embed(target_data) * math.sqrt(self.embed_size)

        if self.vae_setting == True:
            """
            return log_prob, mean, log_var, z
            """
        else:
            
            z = self.encoder(input_embedding, timestamp)
            output = self.decoder(z, target_embedding, input_embedding, timestamp)
            
            return output 



    def get_embedding(self, input_data):
        return self.data_embed(input_data)



class encoderTransformer(nn.Module):
    def __init__(self, args, batch_size, input_size, nhead, latent_size, num_layers, device):
        super(encoderTransformer, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size # input_size == embeding_size
        self.nhead = nhead
        self.latent_size= latent_size
        self.num_layers = num_layers
        self.args =args
        self.device = device

        self.src_mask = None

        self.pos_encoder = positionalEncoding(input_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=input_size*4, batch_first=True, device=device)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

        self.linear_layer1 = nn.Linear(input_size, latent_size * 2)
        self.linear_layer2 = nn.Linear(latent_size * 2, latent_size)

        self.activation_function = nn.Sequential(self.linear_layer1, nn.GELU(), self.linear_layer2)

    def forward(self, input_embedding, timestamp=None, has_mask = True):

        
        if self.args.vae_setting == True: #transformer-based-vae
            if has_mask:
                if self.src_mask is None or self.src_mask.size(0) != len(input_embedding):
                    mask = self.generate_square_subsequent_mask(len(input_embedding)).to(self.device)
                    self.src_mask = mask
            else:
                self.src_mask = None
            src_pad_mask = self.generate_padding_mask(input_embedding,0)
            
            mean, log_var = self.vae_encode(input_embedding, self.src_mask, src_pad_mask, timestamp)
            z = self.reparameterize(mean, log_var)
            return z, log_var, mean
        
        else: #transformer
            if has_mask:
                if self.src_mask is None or self.src_mask.size(0) != len(input_embedding):
                    mask = self.generate_square_subsequent_mask(len(input_embedding)).to(self.device)
                    self.src_mask = mask
            else:
                self.src_mask = None
            src_pad_mask = self.generate_padding_mask(input_embedding,0)
            
            z = self.encoder(input_embedding, self.src_mask, timestamp)

            return z

    def generate_padding_mask(self, data, pad_idx):
            padding_mask = (data == pad_idx)
            return padding_mask

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
            
    def encoder(self, input_embedding, src_mask, timestamp):
        input_embedding = self.pos_encoder(input_embedding, timestamp)
        encoder_output = self.transformer_encoder(input_embedding, src_mask)
        return encoder_output
    """
    def vae_encode(self, input_embedding, src_mask, timestamp):

        return mean, log_var
    """
    def reparameterize(self, mean, log_var):
        batch_size = mean.size(0)
        seq_len = mean.size(1)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, seq_len, self.latent_size]).to(self.device) 
        z = eps * std + mean 
        return z



class decoderTransformer(nn.Module):
    def __init__(self, args, batch_size, input_size, nhead, latent_size, vocab_size, embed_layer, num_layers, device, topk=1):
        super(decoderTransformer, self).__init__()
        self.args = args
        self.batch_size = batch_size
        self.input_size = input_size # same as embed_size
        self.nhead = nhead
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.topk = topk
        self.device = device
        self.vocab_size = vocab_size
        self.embed_layer = embed_layer

        self.pos_encoder = positionalEncoding(input_size)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=input_size*4, batch_first=True, device=device)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        self.linear_hidden1 = nn.Linear(latent_size, input_size // 8)
        self.linear_hidden2 = nn.Linear(input_size // 8, input_size)
        self.linear_vocab1 = nn.Linear(input_size, vocab_size // 8)
        self.linear_vocab2 = nn.Linear(vocab_size // 8, vocab_size)

        self.linear_hidden = nn.Sequential(self.linear_hidden1, nn.GELU(), self.linear_hidden2)
        self.linear_vocab = nn.Sequential(self.linear_vocab1, nn.GELU(), self.linear_vocab2)

    def forward(self, z, target_embedding,  input_embedding, timestamp, mem_mask=None):
        
        batch_size = target_embedding.size(0)

        target_embedding = self.pos_encoder(target_embedding, timestamp)
        
        tgt_mask = self.generate_square_subsequent_mask(len(target_embedding)).to(self.device)
        tgt_pad_mask = self.generate_padding_mask(target_embedding, 0)
        mem_pad_mask = self.generate_padding_mask(input_embedding, 0)
        
        if self.args.vae_setting == True:

            hidden = self.linear_hidden(z) 

            output = self.transformer_decoder(tgt=target_embedding, memory=hidden, tgt_mask=tgt_mask, memory_mask=mem_mask, tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=mem_pad_mask) 
            logits = self.linear_vocab(output) 
            log_prob = F.log_softmax(logits, dim=-1) 

            return log_prob
        
        else: #None_VAE

            output = self.transformer_decoder(tgt=target_embedding, memory=z, tgt_mask=tgt_mask, memory_mask=mem_mask, tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=mem_pad_mask) 
            logits = self.linear_vocab(output) 
            log_prob = F.log_softmax(logits, dim=-1) 
            
            return output, log_prob

    def vae_decode(self, z, mem_pad_mask, timestamp):
         
        batch_size = z.size(0)
        seq_len = z.size(1)

        hidden = self.linear_hidden(z)

        output = torch.ones(batch_size, seq_len).long().to(self.device)

        generated_seq = torch.full((batch_size, 1),  1, dtype=torch.long, device=self.device)

        for i in range(1, seq_len):
            tgt_embedding = self.embed_layer(output[:, :i]) 
            tgt_embedding = self.pos_encoder(tgt_embedding, timestamp) 

            tgt_mask = self.generate_square_subsequent_mask(sz=i)

            decoder_output = self.transformer_decoder(tgt=tgt_embedding, memory=hidden, tgt_mask=tgt_mask, tgt_key_padding_mask=None, memory_key_padding_mask=mem_pad_mask) 
            pred_prob = self.linear_vocab(decoder_output)
    
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
            
            last_generated_token_idx = pred_prob.argmax(dim=-1).unsqueeze(1)
            generated_seq = torch.cat((generated_seq, last_generated_token_idx), dim=-1)
        
        generared_sequence = []
        for each_line in output:
            generared_sequence.append(each_line.tolist())

        return output, generated_seq[:, 1:].contiguous()


    def generate_padding_mask(self, data, pad_idx):

            padding_mask = (data == pad_idx)

            return padding_mask

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(self.device)
        return mask



class positionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=30000):
        super(positionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        p_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        p_encoding[:, 0::2] = torch.sin(position * div_term)
        p_encoding[:, 1::2] = torch.cos(position * div_term)
        p_encoding = p_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('p_encoding', p_encoding)

    def forward(self, x, timestamp = None):
        
        if timestamp is not None:
            x = x + self.p_encoding[:timestamp.size(0), :]
        else :
            x = x + self.p_encoding[:x.size(0),:]
        return self.dropout(x)
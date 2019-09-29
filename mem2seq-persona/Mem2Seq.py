import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import optim
from utils.masked_cross_entropy import sequence_mask, masked_cross_entropy
import random
import numpy as np
from utils.metric import f1_score, bleu
import os
import math
from tqdm import tqdm
from torch.distributions import Categorical

class Mem2Seq(nn.Module):
    def __init__(self, hidden_size, n_layers, max_s, max_r, vocab, load_path, save_path, lr, dr, position, device):
        super(Mem2Seq, self).__init__()
        self.input_size = len(vocab)
        self.output_size = len(vocab)
        self.hidden_size = hidden_size
        self.max_r = max_r ## max responce len        
        self.vocab = vocab
        self.save_path = save_path
        self.lr = lr
        self.n_layers = n_layers
        self.dropout = dr
        self.position = position
        self.device = device
        
        if load_path:
            self.encoder = torch.load(str(load_path)+'/enc.th').to(device)
            self.decoder = torch.load(str(load_path)+'/dec.th').to(device)
            self.encoder.device = device
            self.decoder.device = device
        else:
            self.encoder = EncoderMemNN(vocab, hidden_size, max_s, n_layers, self.dropout, self.position, device).to(device)
            self.decoder = DecoderMemNN(vocab, hidden_size, max_s, n_layers, self.dropout, self.position, device).to(device)
            
        # Initialize optimizers and criterion
        #self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        #self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        #self.encoder_scheduler = lr_scheduler.StepLR(self.encoder_optimizer, step_size=1, gamma=0.5)
        #self.decoder_scheduler = lr_scheduler.StepLR(self.decoder_optimizer, step_size=1, gamma=0.5)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)
        self.loss = 0
        self.loss_ptr = 0
        self.loss_vocab = 0
        self.print_every = 1
        self.batch_size = 0

    def print_loss(self):    
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr =  self.loss_ptr / self.print_every
        print_loss_vocab =  self.loss_vocab / self.print_every
        self.print_every += 1     
        return 'L:{:.2f}, VL:{:.2f}, PL:{:.2f}'.format(print_loss_avg, print_loss_vocab, print_loss_ptr)
    
    def save_model(self, version, epoch):
        directory = self.save_path + str(version) + '/' + str(epoch)
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory+'/enc.th')
        torch.save(self.decoder, directory+'/dec.th')
        
    def train_batch(self, data, clip, teacher_forcing_ratio, reset):  
        input_batches = data[0]
        input_lengths = data[1]
        target_batches = data[2]
        target_lengths = data[3]
        target_index = data[4]
        batch_size = len(data[1])
        
        if reset:
            self.loss = 0
            self.loss_ptr = 0
            self.loss_vocab = 0
            self.print_every = 1

        self.batch_size = batch_size
        # Zero gradients of both optimizers
        #self.encoder_optimizer.zero_grad()
        #self.decoder_optimizer.zero_grad()
        self.optimizer.zero_grad()
        loss_Vocab, loss_Ptr= 0, 0

        # Run words through encoder
        decoder_hidden = self.encoder(input_batches).unsqueeze(0)
        self.decoder.load_memory(input_batches)

        # Prepare input and output variables
        decoder_input = torch.tensor([self.vocab.sos_id] * batch_size, dtype=torch.long, device=self.device)
        
        max_target_length = max(target_lengths)
        all_decoder_outputs_vocab = torch.zeros(max_target_length, batch_size, self.output_size, device=self.device)
        all_decoder_outputs_ptr = torch.zeros(max_target_length, batch_size, input_batches.size(0), device=self.device)

        # Choose whether to use teacher forcing
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        
        if use_teacher_forcing:    
            # Run through decoder one time step at a time
            for t in range(max_target_length):
                decoder_ptr, decoder_vocab, decoder_hidden  = self.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
                all_decoder_outputs_vocab[t] = decoder_vocab
                all_decoder_outputs_ptr[t] = decoder_ptr
                decoder_input = target_batches[t] # Chosen word is next input
        else:
            for t in range(max_target_length):
                decoder_ptr, decoder_vocab, decoder_hidden = self.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
                _, toppi = decoder_ptr.data.topk(1)
                _, topvi = decoder_vocab.data.topk(1)
                all_decoder_outputs_vocab[t] = decoder_vocab
                all_decoder_outputs_ptr[t] = decoder_ptr
                ## get the correspective word in input
                top_ptr_i = torch.gather(input_batches[:, :, 0], 0, toppi.view(1, -1)).transpose(0, 1)
                #next_in = [top_ptr_i[i].item() if (toppi[i].item() < input_lengths[i]-1) else topvi[i].item() for i in range(batch_size)]
                next_in = [top_ptr_i[i] if (toppi[i].item() < input_lengths[i]-1) else topvi[i] for i in range(batch_size)]

                decoder_input = torch.tensor(next_in, dtype=torch.long, device=self.device) # Chosen word is next input
                    
        #Loss calculation and backpropagation
        loss_Vocab = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(), # -> batch x seq
            target_batches.transpose(0, 1).contiguous(), # -> batch x seq
            target_lengths,
            device=self.device
        )
        loss_Ptr = masked_cross_entropy(
            all_decoder_outputs_ptr.transpose(0, 1).contiguous(), # -> batch x seq
            target_index.transpose(0, 1).contiguous(), # -> batch x seq
            target_lengths,
            device=self.device
        )

        loss = loss_Vocab + loss_Ptr
        loss.backward()
        
        # Clip gradient norms
        ec = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        dc = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)
        
        # Update parameters with optimizers
        #self.encoder_optimizer.step()
        #self.decoder_optimizer.step()
        self.optimizer.step()
        self.loss += loss.item()
        self.loss_ptr += loss_Ptr.item()
        self.loss_vocab += loss_Vocab.item()
        
    def generate_batch(self, data, temp=0):
        input_batches = data[0]
        input_lengths = data[1]
        target_batches = data[2]
        target_lengths = data[3]
        target_index = data[4]
        src_plain = data[5]
        batch_size = len(data[1])
        
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)
        # Run words through encoder
        decoder_hidden = self.encoder(input_batches).unsqueeze(0)
        self.decoder.load_memory(input_batches)

        # Prepare input and output variables
        decoder_input = torch.tensor([self.vocab.sos_id] * batch_size, dtype=torch.long, device=self.device)
        
        max_target_length = max(target_lengths)
        all_decoder_outputs_vocab = torch.zeros(self.max_r, batch_size, self.output_size, device=self.device)
        all_decoder_outputs_ptr = torch.zeros(self.max_r, batch_size, input_batches.size(0), device=self.device)

        decoded_words = [[] for i in range(batch_size)]
        gates = [[] for i in range(batch_size)]
        
        p = [[w[0] for w in elm] for elm in src_plain]
        
        # Run through decoder one time step at a time
        for t in range(self.max_r):
            decoder_ptr, decoder_vocab, decoder_hidden = self.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
            _, topvi = decoder_vocab.data.topk(1)
            _, toppi = decoder_ptr.data.topk(1)
            top_ptr_i = torch.gather(input_batches[:, :, 0], 0, toppi.view(1, -1)).transpose(0, 1)
            
            #next_in = [top_ptr_i[i].item() if (toppi[i].item() < input_lengths[i]-1) else topvi[i].item() for i in range(batch_size)]
            next_in = []

            for i in range(batch_size):
                if toppi[i].item() < len(p[i]) - 1:
                    decoded_words[i].append(p[i][toppi[i].item()])
                    next_in.append(top_ptr_i[i])
                    gates[i].append(1)
                else:
                    if temp == 0:
                        ind = topvi[i].item()
                    else:
                        ind = Categorical(F.softmax(decoder_vocab[i] / temp, dim=0)).sample().item()
                    decoded_words[i].append(self.vocab.id2token[ind])
                    next_in.append(ind)
                    gates[i].append(0)
            
            decoder_input = torch.tensor(next_in, dtype=torch.long, device=self.device) # Chosen word is next input
            
            all_decoder_outputs_vocab[t] = decoder_vocab
            all_decoder_outputs_ptr[t] = decoder_ptr
                    
        #Loss calculation
        loss_Vocab = masked_cross_entropy(
            all_decoder_outputs_vocab[:max_target_length].transpose(0, 1).contiguous(), # -> batch x seq
            target_batches.transpose(0, 1).contiguous(), # -> batch x seq
            target_lengths,
            device=self.device
        )
        loss_Ptr = masked_cross_entropy(
            all_decoder_outputs_ptr[:max_target_length].transpose(0, 1).contiguous(), # -> batch x seq
            target_index.transpose(0, 1).contiguous(), # -> batch x seq
            target_lengths,
            device=self.device
        )
        loss_Vocab_masked = masked_cross_entropy(
            all_decoder_outputs_vocab[:max_target_length].transpose(0, 1).contiguous(), # -> batch x seq
            target_batches.transpose(0, 1).contiguous(), # -> batch x seq
            target_lengths,
            device=self.device,
            gates=[[int(not g) for g in gs][:max_target_length] for gs in gates]
        )
        loss_Ptr_masked = masked_cross_entropy(
            all_decoder_outputs_ptr[:max_target_length].transpose(0, 1).contiguous(), # -> batch x seq
            target_index.transpose(0, 1).contiguous(), # -> batch x seq
            target_lengths,
            device=self.device,
            gates=[gs[:max_target_length] for gs in gates]
        )

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)
        
        return decoded_words, loss_Vocab.item(), loss_Ptr.item(), loss_Vocab_masked.item(), loss_Ptr_masked.item()


    def evaluate(self, dev, temp=0):
        f1 = 0.0
        b = 0.0
        l_v = 0.0
        l_p = 0.0
        l_v_m = 0.0
        l_p_m = 0.0
        len_total = 0
        pbar = tqdm(enumerate(dev), total=len(dev))
        for _, data_dev in pbar: 
            w_batch, l_v_batch, l_p_batch, l_v_m_batch, l_p_m_batch = self.generate_batch(data_dev, temp)
            for j, words in enumerate(w_batch):
                for i, word in enumerate(words):
                    if word == '<eos>' and i < len(words) - 1:
                        words = words[:i+1]
                        break
                if len(words) < 1:
                    continue
                b += bleu(words, data_dev[6][j])
                f1 += f1_score(words, data_dev[6][j])
                
            len_total += len(w_batch)
            l_v += l_v_batch * len(w_batch)
            l_p += l_p_batch * len(w_batch)
            l_v_m += l_v_m_batch * len(w_batch)
            l_p_m += l_p_m_batch * len(w_batch)
            
            pbar.set_description('TL: {:.3f}, VL: {:.3f}, PL: {:.3f}, VML: {:.3f}, PML: {:.3f}'.format((l_v + l_p) / len_total, l_v / len_total, l_p / len_total, l_v_m / len_total, l_p_m / len_total))
            
        perplexity = math.exp((l_v + l_p) / len_total)
        perplexity_m = math.exp((l_v_m + l_p_m) / len_total)
        
        return (l_v + l_p) / len_total, b / len_total, f1 / len_total, perplexity, perplexity_m


class EncoderMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, max_s, hop, dropout, position, device):
        super(EncoderMemNN, self).__init__()
        self.num_vocab = len(vocab)
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.position = position
        self.pad_id = vocab.pad_id
        
        if position:
            self.pos_embedding = PositionalEncoder(embedding_dim, max_s, self.pad_id)
            
        for hop in range(self.max_hops+1):
            C = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=self.pad_id)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)
        self.device = device
        
    def get_state(self, bsz):
        """Get cell states and hidden states."""
        return torch.zeros(bsz, self.embedding_dim, device=self.device)


    def forward(self, story):
        story = story.transpose(0, 1)
        
        # Dropout.
        if self.training:
            ones = np.ones(story.size())
            rand_mask = np.random.binomial([np.ones((story.size(0), story.size(1)))], 1-self.dropout)[0]
            ones[:, :, 0] = ones[:, :, 0] * rand_mask
            story = story * torch.tensor(ones, dtype=torch.long, device=self.device)
        
        # Positional embedding.
        if self.position:
            embed_pos = self.pos_embedding(story)
        
        u = [self.get_state(story.size(0))]
        for hop in range(self.max_hops):
            # Input memory.
            embed_A = self.C[hop](story.view(story.size(0), -1)) # b * (m * s) * e
            embed_A = embed_A.view(story.size() + (embed_A.size(-1), )) # b * m * s * e
            m_A = torch.sum(embed_A, 2).squeeze(2) # b * m * e
            if self.position and hop == 0:
                m_A = m_A + embed_pos
            
            # Attention.
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob = self.softmax(torch.sum(m_A*u_temp, 2))  
            
            # Output memory.
            embed_C = self.C[hop+1](story.view(story.size(0), -1)) # b * (m * s) * e
            embed_C = embed_C.view(story.size() + (embed_C.size(-1), )) # b * m * s * e
            m_C = torch.sum(embed_C, 2).squeeze(2) # b * m * e
            
            # Weighted sum.
            prob = prob.unsqueeze(2).expand_as(m_C)
            o_k  = torch.sum(m_C*prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
            
        return u_k

class DecoderMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, max_s, hop, dropout, position, device):
        super(DecoderMemNN, self).__init__()
        self.num_vocab = len(vocab)
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.position = position
        self.pad_id = vocab.pad_id
        
        if position:
            self.pos_embedding = PositionalEncoder(embedding_dim, max_s, self.pad_id)
        
        for hop in range(self.max_hops+1):
            C = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=self.pad_id)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)
        self.W = nn.Linear(embedding_dim, 1)
        self.W1 = nn.Linear(2*embedding_dim, self.num_vocab)
        self.gru = nn.GRU(embedding_dim, embedding_dim)
        self.device = device

    def load_memory(self, story):
        story = story.transpose(0, 1)
        
        # Dropout.
        if self.training:
            ones = np.ones(story.size())
            rand_mask = np.random.binomial([np.ones((story.size(0), story.size(1)))], 1-self.dropout)[0]
            ones[:, :, 0] = ones[:, :, 0] * rand_mask
            story = story * torch.tensor(ones, dtype=torch.long, device=self.device)
            
        # Positional embedding.
        if self.position:
            embed_pos = self.pos_embedding(story)
            
        self.m_story = []
        for hop in range(self.max_hops+1):
            embed_A = self.C[hop](story.view(story.size(0), -1)) # b * (m * s) * e
            embed_A = embed_A.view(story.size() + (embed_A.size(-1), )) # b * m * s * e
            m_A = torch.sum(embed_A, 2).squeeze(2) # b * m * e
            if self.position and hop == 0:
                m_A = m_A + embed_pos
            self.m_story.append(m_A)

    def ptrMemDecoder(self, enc_query, last_hidden):
        embed_q = self.C[0](enc_query) # b * e
        output, hidden = self.gru(embed_q.unsqueeze(0), last_hidden)
        temp = []
        u = [hidden[0].squeeze()]   
        for hop in range(self.max_hops):
            m_A = self.m_story[hop]
            if(len(list(u[-1].size()))==1): u[-1] = u[-1].unsqueeze(0) ## used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob_lg = torch.sum(m_A*u_temp, 2)
            prob_   = self.softmax(prob_lg)
            m_C = self.m_story[hop+1]
            temp.append(prob_)
            prob = prob_.unsqueeze(2).expand_as(m_C)
            o_k  = torch.sum(m_C*prob, 1)
            if hop == 0:
                p_vocab = self.W1(torch.cat((u[0], o_k), 1))
            u_k = u[-1] + o_k
            u.append(u_k)
        p_ptr = prob_lg 
        return p_ptr, p_vocab, hidden


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

    
class PositionalEncoder(torch.nn.Module):
    def __init__(self, emb_dim, max_len, pad_id):
        """
        Modified Sinusoidal Positional Encoder from
        https://github.com/kaushalshetty/Positional-Encoding/blob/master/encoder.py
        """    
        super(PositionalEncoder,self).__init__()
        self.pad_id = pad_id
        n_position = max_len + 1
        self.position_enc = torch.nn.Embedding(n_position, emb_dim, padding_idx=0)
        self.position_enc.weight.data = self.position_encoding_init(n_position, emb_dim)
        
    def position_encoding_init(self, n_position, emb_dim):
        ''' Init the sinusoid position encoding table '''
        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
            if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # apply sin on 0th,2nd,4th...emb_dim
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # apply cos on 1st,3rd,5th...emb_dim
        return torch.from_numpy(position_enc).type(torch.FloatTensor)    
    
    def get_absolute_pos(self, story):
        padding_mask = story[:, :, 0].eq(self.pad_id)
        positions = torch.cumsum(~padding_mask, dim=-1, dtype=torch.long)
        positions.masked_fill_(padding_mask, 0)
        return positions
        
    def forward(self, story):
        word_pos = self.get_absolute_pos(story)
        word_pos_encoded = self.position_enc(word_pos)
        return word_pos_encoded
        

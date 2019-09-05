import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
from utils.masked_cross_entropy import *
import random
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib
import os
import logging
from tqdm import tqdm

class Mem2Seq(nn.Module):
    def __init__(self, hidden_size, pos_size, max_r, vocab, path, lr, n_layers, dropout, unk_mask, device):
        super(Mem2Seq, self).__init__()
        self.name = "Mem2Seq"
        self.input_size = len(vocab)
        self.output_size = len(vocab)
        self.hidden_size = hidden_size
        self.pos_size = pos_size
        self.max_r = max_r ## max responce len        
        self.vocab = vocab
        self.lr = lr
        self.n_layers = n_layers
        self.dropout = dropout
        self.unk_mask = unk_mask
        self.device = device
        
        if path:
            logging.info("MODEL {} LOADED".format(str(path)))
            self.encoder = torch.load(str(path)+'/enc.th')
            self.decoder = torch.load(str(path)+'/dec.th')
        else:
            self.encoder = EncoderMemNN(vocab, hidden_size, pos_size, n_layers, self.dropout, self.unk_mask, device).to(device)
            self.decoder = DecoderMemNN(vocab, hidden_size, pos_size, n_layers, self.dropout, self.unk_mask, device).to(device)
            
        # Initialize optimizers and criterion
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer,mode='max',factor=0.5,patience=1,min_lr=1e-6, verbose=True)
        self.criterion = nn.MSELoss()
        self.loss = 0
        self.loss_ptr = 0
        self.loss_vac = 0
        self.print_every = 1
        self.batch_size = 0

    def print_loss(self):    
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr =  self.loss_ptr / self.print_every
        print_loss_vac =  self.loss_vac / self.print_every
        self.print_every += 1     
        return 'L:{:.2f}, VL:{:.2f}, PL:{:.2f}'.format(print_loss_avg, print_loss_vac, print_loss_ptr)
    
    def save_model(self, version):
        directory = 'save/' + 'HDD' + str(self.hidden_size) + 'POS' + str(self.pos_size) + 'DR' + str(self.dropout) + 'L' + str(self.n_layers) + 'lr' + str(self.lr) + 'v' + str(version)
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory+'/enc.th')
        torch.save(self.decoder, directory+'/dec.th')
        
    def train_batch(self, input_batches, input_lengths, target_batches, 
                    target_lengths, target_index, batch_size, clip,
                    teacher_forcing_ratio, reset):  

        if reset:
            self.loss = 0
            self.loss_ptr = 0
            self.loss_vac = 0
            self.print_every = 1

        self.batch_size = batch_size
        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
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
                decoder_ptr, decoder_vacab, decoder_hidden  = self.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
                all_decoder_outputs_vocab[t] = decoder_vacab
                all_decoder_outputs_ptr[t] = decoder_ptr
                decoder_input = target_batches[t] # Chosen word is next input
        else:
            for t in range(max_target_length):
                decoder_ptr, decoder_vacab, decoder_hidden = self.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
                _, toppi = decoder_ptr.data.topk(1)
                _, topvi = decoder_vacab.data.topk(1)
                all_decoder_outputs_vocab[t] = decoder_vacab
                all_decoder_outputs_ptr[t] = decoder_ptr
                ## get the correspective word in input
                top_ptr_i = torch.gather(input_batches[:, :, 0], 0, toppi.view(1, -1)).transpose(0, 1)
                next_in = [top_ptr_i[i].item() if (toppi[i].item() < input_lengths[i]-1) else topvi[i].item() for i in range(batch_size)]

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
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.loss += loss.item()
        self.loss_ptr += loss_Ptr.item()
        self.loss_vac += loss_Vocab.item()
        
    def generate_batch(self, batch_size, input_batches, input_lengths, src_plain):
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)
        # Run words through encoder
        decoder_hidden = self.encoder(input_batches).unsqueeze(0)
        self.decoder.load_memory(input_batches)

        # Prepare input and output variables
        decoder_input = torch.tensor([self.vocab.sos_id] * batch_size, dtype=torch.long, device=self.device)

        decoded_words = [[] for i in range(batch_size)]
        
        p = [[w[0] for w in elm] for elm in src_plain]
        
        # Run through decoder one time step at a time
        for t in range(self.max_r):
            decoder_ptr, decoder_vocab, decoder_hidden = self.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
            _, topvi = decoder_vocab.data.topk(1)
            _, toppi = decoder_ptr.data.topk(1)
            top_ptr_i = torch.gather(input_batches[:, :, 0], 0, toppi.view(1, -1)).transpose(0, 1)
            
            next_in = [top_ptr_i[i].item() if (toppi[i].item() < input_lengths[i]-1) else topvi[i].item() for i in range(batch_size)]
            decoder_input = torch.tensor(next_in, dtype=torch.long, device=self.device) # Chosen word is next input

            for i in range(batch_size):
                if toppi[i].item() < len(p[i]) - 1:
                    decoded_words[i].append(p[i][toppi[i].item()])
                else:
                    ind = topvi[i].item()
                    decoded_words[i].append(self.vocab.id2token[ind])

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)
        
        return decoded_words


    def evaluate(self, dev):
        logging.info("STARTING EVALUATION")
        cc = SmoothingFunction()
        bleu = 0.0
        len_total = 0
        pbar = tqdm(enumerate(dev), total=len(dev))
        for _, data_dev in pbar: 
            words_batch = self.generate_batch(len(data_dev[1]), data_dev[0], data_dev[1], data_dev[5])
            for j, words in enumerate(words_batch):
                for i, word in enumerate(words):
                    if word == '<eos>' and i < len(words) - 1:
                        words = words[:i+1]
                        break
                if len(words) < 1:
                    continue
                bleu += sentence_bleu([data_dev[6][j]], words, smoothing_function=cc.method4)
            len_total += len(words_batch)
            pbar.set_description('BLEU: {:.3f}'.format(bleu / len_total))
        
        return bleu / len_total


class EncoderMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, pos_embedding_dim, hop, dropout, unk_mask, device):
        super(EncoderMemNN, self).__init__()
        self.num_vocab = len(vocab)
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.unk_mask = unk_mask
        self.pad_id = vocab.pad_id
        
        self.pos_embedding = nn.Embedding(pos_embedding_dim + 1, embedding_dim, padding_idx=self.pad_id)
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
        if self.unk_mask and self.training:
            ones = np.ones(story.size())
            rand_mask = np.random.binomial([np.ones((story.size(0), story.size(1)))], 1-self.dropout)[0]
            ones[:, :, 0] = ones[:, :, 0] * rand_mask
            story = story * torch.tensor(ones, dtype=torch.long, device=self.device)
        
        # Positional embedding.
        padding_mask = story[:, :, 0].eq(self.pad_id)
        positions = torch.cumsum(~padding_mask, dim=-1, dtype=torch.long)
        positions.masked_fill_(padding_mask, self.pad_id)
        embed_pos = self.pos_embedding(positions)
        
        u = [self.get_state(story.size(0))]
        for hop in range(self.max_hops):
            # Input memory.
            embed_A = self.C[hop](story.view(story.size(0), -1)) # b * (m * s) * e
            embed_A = embed_A.view(story.size() + (embed_A.size(-1), )) # b * m * s * e
            m_A = torch.sum(embed_A, 2).squeeze(2) # b * m * e
            m_A = m_A + embed_pos
            
            # Attention.
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob = self.softmax(torch.sum(m_A*u_temp, 2))  
            
            # Output memory.
            embed_C = self.C[hop+1](story.view(story.size(0), -1)) # b * (m * s) * e
            embed_C = embed_C.view(story.size() + (embed_C.size(-1), )) # b * m * s * e
            m_C = torch.sum(embed_C, 2).squeeze(2) # b * m * e
            m_C = m_C + embed_pos
            
            # Weighted sum.
            prob = prob.unsqueeze(2).expand_as(m_C)
            o_k  = torch.sum(m_C*prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
            
        return u_k

class DecoderMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, pos_embedding_dim, hop, dropout, unk_mask, device):
        super(DecoderMemNN, self).__init__()
        self.num_vocab = len(vocab)
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.unk_mask = unk_mask
        self.pad_id = vocab.pad_id
        
        #self.pos_embedding = nn.Embedding(pos_embedding_dim + 1, embedding_dim, padding_idx=self.pad_id)
        for hop in range(self.max_hops+1):
            C = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=self.pad_id)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)
        self.W = nn.Linear(embedding_dim, 1)
        self.W1 = nn.Linear(2*embedding_dim,self.num_vocab)
        self.gru = nn.GRU(embedding_dim, embedding_dim)
        self.device = device

    def load_memory(self, story):
        story = story.transpose(0, 1)
        
        # Dropout.
        if self.unk_mask and self.training:
            ones = np.ones(story.size())
            rand_mask = np.random.binomial([np.ones((story.size(0), story.size(1)))], 1-self.dropout)[0]
            ones[:, :, 0] = ones[:, :, 0] * rand_mask
            story = story * torch.tensor(ones, dtype=torch.long, device=self.device)
            
        # Positional embedding.
        #padding_mask = story[:, :, 0].eq(self.pad_id)
        #positions = torch.cumsum(~padding_mask, dim=-1, dtype=torch.long)
        #positions.masked_fill_(padding_mask, self.pad_id)
        #embed_pos = self.pos_embedding(positions)
            
        self.m_story = []
        for hop in range(self.max_hops+1):
            embed_A = self.C[hop](story.view(story.size(0), -1)) # b * (m * s) * e
            embed_A = embed_A.view(story.size() + (embed_A.size(-1), )) # b * m * s * e
            m_A = torch.sum(embed_A, 2).squeeze(2) # b * m * e
            #m_A = m_A + embed_pos
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
            if (hop==0):
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

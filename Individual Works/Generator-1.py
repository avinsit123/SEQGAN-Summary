#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 18:08:56 2019

@author: r17935avinash
"""

from torchtext.data import Field
import pandas as pd
from torch import nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import random
import math
import time
import numpy as np

#torch.manual_seed(16)
#torch.backends.cudnn.enabled = False 

#!git clone https://github.com/IBM/pytorch-seq2seq.git
tokenize = lambda x: x.split()
TEXT = Field(sequential=True,tokenize=tokenize,init_token = '<sos>', 
            eos_token = '<eos>',lower=True)
SUMMARY = Field(sequential=True,tokenize=tokenize,init_token = '<sos>', 
            eos_token = '<eos>',lower=True)
from torchtext.data import TabularDataset
tv_datafields = [("headlines",SUMMARY),("text",TEXT)]

#ass = pd.read_csv("/media/data_dump_1/avinash/news_summary_more.csv")
#print(ass.head(1))
trn = TabularDataset(
    path="/media/data_dump_1/avinash/news_summary_more.csv",
       format='csv',
        skip_header=True, 
        fields=tv_datafields)
 
TEXT.build_vocab(trn,max_size=90000,min_freq=2)
SUMMARY.build_vocab(trn,max_size=60000)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

train_iter = BucketIterator(
        trn,
        batch_size=64,
        device=device,
)

SRC_VOCAB_SIZE = len(TEXT.vocab)
TGT_VOCAB_SIZE = len(SUMMARY.vocab)
MIN_SEQ_LEN = 14
MAX_SEQ_LEN = 24
BATCH_SIZE = 64
MLE_TRAIN_EPOCHS = 15
ADV_TRAIN_EPOCHS = 50
DIS_TRAIN_EPOCHS = 2

GEN_EMBEDDING_DIM = 150
GEN_HIDDEN_DIM = 250
DIS_EMBEDDING_DIM = 128
DIS_HIDDEN_DIM = 128

CAPACITY_RM = 100000
PRETRAIN_GENERATOR = False
PRETRAIN_DISCRIMINATOR = False
POLICY_GRADIENT = True
ACTOR_CHECKPOINT = "generator_checkpoint19.pth.tar"
DISCRIMINATOR_MLE_LR = 5e-2
ACTOR_LR = 1e-2
CRITIC_LR = 1e-2
DISCRIMINATOR_LR = 1e-2
AC = True
SEQGAN = True
if SEQGAN:
    DISCRIMINATOR_CHECKPOINT = "discriminator_final.pth.tar"
else:
    DISCRIMINATOR_CHECKPOINT = None#"discriminator_final_LM2.pth.tar"
AC_WARMUP = 1000
DISCOUNT_FACTOR = 0.99
BATCH_SIZE_TESTING = 256
NUM_SAMPLES = 3

#print(len(TEXT.vocab),len(SUMMARY.vocab),"HELLO")

    
from EncoderRNN import EncoderRNN
from DecoderRNN import DecoderRNN
from Seq2Seq import Seq2seq
from torch.nn.utils import clip_grad_norm_

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:1')
else:
    DEVICE = torch.device('cpu')  #'cuda:0'

class Generator(nn.Module):
    def __init__(
            self,
            sos_id,
            eou_id,
            src_vocab_size,
            tgt_vocab_size,
            hidden_size,
            embed_size,
            max_len,
            beam_size=3,
            enc_n_layers=2,
            enc_dropout=0.2,
            enc_bidirectional=True,
            dec_n_layers=2,
            dec_dropout=0.2,
            dec_bidirectional=True,
            teacher_forcing_ratio=0):
        super(Generator, self).__init__()

        self.sos_id = sos_id
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.max_len = max_len

        self.encoder = EncoderRNN(src_vocab_size, max_len-1, hidden_size, 0, enc_dropout, enc_n_layers, True, 'gru', False, None)
        self.decoder = DecoderRNN(tgt_vocab_size, max_len-1, hidden_size*2 if dec_bidirectional else hidden_size, sos_id, eou_id, dec_n_layers, 'gru', dec_bidirectional, 0, dec_dropout, True)
        # self.beam_decoder = TopKDecoder(self.decoder, beam_size)
        self.seq2seq = Seq2seq(self.encoder, self.decoder)

    def sample(self, src, tgt, TF=0.5):
        sentences, probabilities, hiddens = self.seq2seq(src, target_variable=tgt, teacher_forcing_ratio=TF, sample=True)
        #print("dgsh")
        return sentences, probabilities, hiddens

    def forward(self, src, tgt, hack=False):
        src = src.t()
        tgt = tgt.t()
        
        outputs, _, meta_data = self.seq2seq(src, target_variable=tgt, teacher_forcing_ratio=self.teacher_forcing_ratio)

        batch_size = outputs[0].size(0)
        start_tokens = torch.zeros(batch_size, self.tgt_vocab_size, device=outputs[0].device)
        start_tokens[:,self.sos_id] = 1

        outputs = [start_tokens] + outputs
        outputs = torch.stack(outputs)
        if hack == True:
            return outputs, meta_data
        return outputs

        # NOTICE THAT DISCOUNT FACTOR is 1
    def compute_reinforce_loss(self, rewards, probabilities):
        rewards = rewards.to(DEVICE)
        probabilities = probabilities.to(DEVICE)
        sentences_level_reward = torch.mean(rewards, 1)
        R_s_w = rewards
        
        
        sent_len = rewards.size(1)
        J = 0
        for k in range(sent_len):
            R_k = torch.sum(R_s_w[:,k:], 1)
            prob = probabilities[:,k]
            J = (J + R_k*prob)*(DISCOUNT_FACTOR)

        loss = -torch.mean(J)
        return loss

    def try_get_state_dicts(self,directory='./', prefix='generator_checkpoint', postfix='.pth.tar'):
        files = os.listdir(directory)
        files = [f for f in files if f.startswith(prefix)]
        files = [f for f in files if f.endswith(postfix)]

        epoch_nums = []
        for file in files:
            number = file[len(prefix):-len(postfix)]
            try:
                epoch_nums.append(int(number))
            except:
                pass

        if len(epoch_nums) < 2:
            return None

        last_complete_epoch = sorted(epoch_nums)[-2]
        filename = prefix + str(last_complete_epoch) + postfix

        data = torch.load(filename)
        return data

    def train_generator_MLE_batch(self, context, reply, optimizer, pad_id):
        context = context.t()
        reply = reply.t()
        loss_func = torch.nn.NLLLoss(ignore_index=pad_id) # TO DEVICE?
        output = self.forward(context, reply)
        pred_dist = output[1:].view(-1, self.tgt_vocab_size)
        tgt_tokens = reply[1:].contiguous().view(-1)
        loss = loss_func(pred_dist, tgt_tokens)

        # Backpropagate loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 10) # might be something to check
        optimizer.step()

    def train_generator_MLE(self, optimizer, data_loader, vld_data_loader, epochs, device):

        pad_id = TEXT.vocab.stoi['<pad>']

        loss_func = torch.nn.NLLLoss(ignore_index=pad_id)
        loss_func.to(device)

        start_epoch = 0
        # saved_data = try_get_state_dicts()
        # if saved_data is not None:
        #     start_epoch = saved_data['epoch']
        #     self.load_state_dict(saved_data['state_dict'])
        #     optimizer.load_state_dict(saved_data['optimizer'])

        loss_per_epoch = []
        for epoch in range(start_epoch, epochs):
            print('epoch %d : ' % (epoch + 1))

            total_loss = 0
            losses = []
            for (iters, result) in enumerate(data_loader):
                optimizer.zero_grad()
                context = result.text
                reply = result.headlines
                context = context.to(device)
                reply = reply.to(device)
                
                output = self.forward(context, reply)
                
                #print(output.size())
                # Compute loss
                pred_dist = output[1:].view(-1, self.tgt_vocab_size)
                tgt_tokens = reply[1:].contiguous().view(-1)

                loss = loss_func(pred_dist, tgt_tokens)

                # Backpropagate loss
                loss.backward()
                clip_grad_norm_(self.parameters(), 10)
                optimizer.step()
                total_loss += loss.data.item()
                losses.append(loss)
                
                #print(total_loss)

                # Print updates
                if iters % 25 == 0 and iters != 0:
                    print('[Epoch {} iter {}] loss: {}'.format(epoch,iter,total_loss/25))
                    total_loss = 0
                    torch.save({
                        'epoch': epoch+1,
                        'state_dict': self.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'loss'      : losses,
                    },'generator_checkpoint12{}.pth.tar'.format(epoch))
              

            loss_per_epoch.append(total_loss)
        torch.save(loss_per_epoch, "generator_final_loss.pth.tar")
        return losses

    def monte_carlo(self, dis, context, reply, hiddens, num_samples):

        """
        Samples the network using a batch of source input sequence. Passes these inputs
        through the decoder and instead of taking the top1 (like in forward), sample
        using the distribution over the vocabulary
        Inputs: start of sequence, maximum sample sequence length and num of samples
        Outputs: samples
        samples: num_samples x max_seq_length (a sampled sequence in each row)
        Inputs: dialogue context (and maximum sample sequence length
        Outputs: samples
            - samples: batch_size x reply_length x num_samples x max_seq_length"""

        # Initialize sample
        batch_size = reply.size(0)
        vocab_size = self.decoder.output_size
        # samples_prob = torch.zeros(batch_size, self.max_len)
        encoder_output, _ = self.encoder(context)
        rewards = torch.zeros(reply.size(1), num_samples, batch_size)
        function = F.log_softmax
        reply = reply.to(DEVICE)
        #print(reply.size())
        for t in range(reply.size(1)):
            # Hidden state from orignal generated sequence until t
            for n in range(num_samples):
                samples = reply.clone()
                hidden = hiddens[t].to(DEVICE)
                output = reply[:,t].to(DEVICE)
                # samples_prob[:,0] = torch.ones(output.size())
                # Pass through decoder and sample from resulting vocab distribution
                for next_t in range(t+1, samples.size(1)):
                    decoder_output, hidden, step_attn = self.decoder.forward_step(output.reshape(-1, 1).long(), hidden, encoder_output,
                                                                             function=function)
                    # Sample token for entire batch from predicted vocab distribution
                    decoder_output = decoder_output.reshape(batch_size, self.tgt_vocab_size).detach()
                    batch_token_sample = torch.multinomial(torch.exp(decoder_output), 1).view(-1)
                    # prob = torch.exp(decoder_output)[np.arange(batch_size), batch_token_sample]
                    # samples_prob[:, next_t] = prob
                    samples[:, next_t] = batch_token_sample
                    output = batch_token_sample
                reward = dis.batchClassify(samples.long().to(DEVICE), context.long().to(DEVICE)).detach() ## FIX CONTENT

                rewards[t, n, :] = reward
                
        reward_per_word = torch.mean(rewards, dim=1).permute(1, 0)
        return reward_per_word
    
SOS = TEXT.vocab.stoi['<sos>']
EOU = TEXT.vocab.stoi['<eos>']
PAD = TEXT.vocab.stoi['<pad>']



    
SOS = TEXT.vocab.stoi['<sos>']
EOU = TEXT.vocab.stoi['<eos>']
PAD = TEXT.vocab.stoi['<pad>']

gen = Generator(SOS,EOU,SRC_VOCAB_SIZE,TGT_VOCAB_SIZE, GEN_HIDDEN_DIM, GEN_EMBEDDING_DIM, MAX_SEQ_LEN)
gen = gen.to(DEVICE)
genMLE_optimizer = optim.Adam(gen.parameters(), lr = 0.00001)
chkpt = torch.load("generator_checkpoint_changed_against1.pth.tar")
gen.load_state_dict(chkpt["state_dict"])
#gen.train_generator_MLE(genMLE_optimizer, train_iter, MLE_TRAIN_EPOCHS,DEVICE)

print("generator_checkpoint_changed_against1.pth.tar")

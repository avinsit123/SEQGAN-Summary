#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 22:03:44 2019

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

################### TRAIN GENERATOR USING SAMPLES FROM GENERATOR AND ORIGINAL TEXT ######################################################
import torch
import torch.autograd as autograd
import torch.nn as nn
import pdb
import time

class Discriminator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False, dropout=0.2, device='cpu'):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.device = device

        ## Reply embedding
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru2hidden = nn.Linear(2*2*hidden_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)

        # context embedding
        self.embeddings2 = nn.Embedding(vocab_size, embedding_dim)
        self.gru2 = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru2hidden2 = nn.Linear(2*2*hidden_dim, hidden_dim)
        self.dropout_linear2 = nn.Dropout(p=dropout)

        self.hidden2out = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2*2*1, batch_size, self.hidden_dim)).to(self.device)
        return h

    def forward(self, reply, context, hidden, hidden2):
        # REPLy dim  
        # batch_size x seq_len
        emb = self.embeddings(reply)                               # batch_size x seq_len x embedding_dim
        emb = emb.permute(1, 0, 2)                                 # seq_len x batch_size x embedding_dim
        _, hidden = self.gru(emb, hidden)                          # 4 x batch_size x hidden_dim
        hidden = hidden.permute(1, 0, 2).contiguous()              # batch_size x 4 x hidden_dim
        out = self.gru2hidden(hidden.view(-1, 4*self.hidden_dim))
        # batch_size x 4*hidden_dim
        #print(out.size(),"sadfg")
        out = torch.tanh(out)
        out_reply = self.dropout_linear(out)
        out = self.hidden2out(out_reply) # batch_size x 1
        out = torch.sigmoid(out)
        #print(out.size(),"sfgd")
        return out

    def batchClassify(self, reply, context):
        """
        Classifies a batch of sequences.
        Inputs: inp
            - inp: batch_size x seq_len
        Returns: out
            - out: batch_size ([0,1] score)
        """

        h = self.init_hidden(reply.size()[0])
        h2 = self.init_hidden(context.size()[1])
        out = self.forward(reply.long(), context.long(), h, h2)
        return out.view(-1)

    def batchBCELoss(self, inp, target):
        """
        Returns Binary Cross Entropy Loss for discriminator.
         Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size (binary 1/0)
        """

        loss_fn = nn.BCELoss()
        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)
        return loss_fn(out, target)
    
def pre_train_discriminator(dis, dis_opt, gen, epochs,train_iter):

    """Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs."""


    start_epoch = 0
    loss_per_epoch = []
    losses = []
    real_list = []
    fake_list = []
    count = 0
    print("Number of epochs", epochs)
    for epoch in range(start_epoch, epochs):
        print('epoch %d : ' % (epoch + 1))
        
        total_loss = 0
        negative_loss = 10000000
        loss = nn.BCELoss()
        gf = 0
        #print("asf")
        for (iter,result) in enumerate(train_iter):
            #print(result)
            context = result.text
            real_reply = result.headlines
            
            
            context = context.to(DEVICE)
            real_reply = real_reply.to(DEVICE)

            dis_opt.zero_grad()
            
            #print(real_reply.size(),"sdfsdg")
            # Sample setences
            #print("Hi is it good")
            with torch.no_grad():
                fake_reply, _, _ = gen.sample(context.t(), real_reply.t())
 
            #print("asf")
            # Add padding
            fake_reply = fill_with_padding(fake_reply, EOU, PAD).detach()
           # print(fake_reply.size(),"sdfsdg")
            real_reply =real_reply.t()
             #####

            if SEQGAN:
                
                #print(fake_reply.size(),real_reply.size())
                #fake_labels = torch.from_numpy(np.zeros(fake_reply.size(0))).float().to(DEVICE)
                #real_labels = torch.from_numpy(np.ones(real_reply.size(0))).float().to(DEVICE)
                
                fake_labels = torch.from_numpy(np.random.uniform(0, 0.3, size=(fake_reply.size(0)))).float().to(DEVICE)
                real_labels = torch.from_numpy(np.random.uniform(0.7, 1.2, size=(fake_reply.size(0)))).float().to(DEVICE)                

                # Get probabilities/rewards for real/fake
                real_r = dis.batchClassify(real_reply, context)
                fake_r = dis.batchClassify(fake_reply.to(DEVICE), context)

                # Learn with fake_r

                loss_fake = loss(fake_r, fake_labels)

                loss_real = loss(real_r, real_labels)
                loss_total = loss_real + loss_fake
                loss_total.backward()
                losses.append(loss_total.item())
                
                gf+=1;
                if(gf%25==0):
                  if(gf%100==0):
                      print(np.average(real_r.cpu().detach().numpy()),np.average(fake_r.cpu().detach().numpy()))
                  print("The Loss is ",loss_total)
                  torch.save({
                        'epoch': epoch+1,
                        'state_dict': dis.state_dict(),
                        'optimizer' : dis_opt.state_dict()
                    },'discriminator_checkpointlr{}.pth.tar'.format(epoch))
                  
             ####
           
            else:
                rewards_real, sentence_level_rewards_real = dis.get_rewards(real_reply.to(DEVICE), PAD)
                rewards, sentence_level_rewards_fake = dis.get_rewards(fake_reply.long().to(DEVICE), PAD)
                
                #print(rewards_real[0],"asaf")
                #print(rewards[0],"sfgs")
                #print(rewards.size(),sentence_level_rewards_fake.size(),"aaaad")
                
                #print("Sentence_level_rewards_real:",sentence_level_rewards_real)
                #print("sentence_level_rewards_fake:",sentence_level_rewards_fake)
                real_list.append(torch.mean(sentence_level_rewards_real).item())
                fake_list.append(torch.mean(sentence_level_rewards_fake).item())

                rewards_real = Discount(rewards_real,0.99)
                rewards = Discount(rewards,0.99)
                
                avg_real_rewards = torch.mean(rewards_real,dim=1)
                avg_fake_rewards = torch.mean(rewards,dim=1)
                
                #print(avg_real_rewards.size())
                
                loss_real = torch.mean(avg_real_rewards*sentence_level_rewards_real)
                loss_fake = torch.mean(avg_fake_rewards*sentence_level_rewards_fake)
                
                
                #print(sentence_level_rewards_fake,"sfegw")
                #print(sentence_level_rewards_real,"sfewgw")
                
                #print(real_list,"sfg")
                total_loss =  -1 * (loss_real - loss_fake)
                
                gf+=1;
                if(gf%25==0):
                    if gf%100==0:
                        print(rewards_real[0],fake_rewards[0])
                    print("The Loss is ",total_loss)
                    torch.save({
                        'epoch': epoch+1,
                        'state_dict': dis.state_dict(),
                        'optimizer' : dis_opt.state_dict()
                    },'discriminator_checkpointlr{}.pth.tar'.format(epoch))
                  
                total_loss.backward()
            dis_opt.step()
   
  

    # smooth results
    real = []
    fake = []
    interval = 20
    for i in range(len(real_list)):
        if i % interval == 0:
            real_mean = np.mean(real_list[i:i+interval])
            fake_mean = np.mean(fake_list[i:i+interval])
            print("real mean ", real_mean)
            print("fake mean ", fake_mean)
            real.append(real_mean)
            fake.append(fake_mean)

    plt.figure(1)
    plt.plot(real, label='real')
    plt.plot(fake, label='fake')
    plt.ylabel('Reward')
    plt.xlabel('Iterations x'+ str(interval))
    plt.legend()
    plt.savefig('rewards.png')

    torch.save(dis.state_dict(), "discriminator_final.pth.tar")
    plt.figure(2)
    plt.plot(losses)
    plt.ylabel("Loss")
    plt.xlabel("iterations x "+ str(interval))
    plt.savefig("loss_disc_pretrain.png")
    
def fill_with_padding(sentences, u_token, pad_token):
    """
    Takes a batch of sentences with equal lengths as input.
    Returns same size of batch but with padding filling after the first
    end of utterence token.
    """

    for i in range(sentences.size(0)):
        sent = sentences[i]
        idx = (sent == u_token).nonzero()
        if len(idx) > 0:
            idx = idx[0].item()
            split = torch.split(sent, idx+1)[0].to(DEVICE)
            padding = pad_token * torch.ones(sentences.size(1) - len(split))
            padding = padding.to(DEVICE)
            pad_sent = torch.cat((split, padding))
            sentences[i][:] = pad_sent
    return sentences

dis = Discriminator(DIS_EMBEDDING_DIM,DIS_HIDDEN_DIM,TGT_VOCAB_SIZE,MAX_SEQ_LEN,device=DEVICE)
dis = dis.to(device)
lr = 0.001
beta1 = 0.42
beta2 = 0.999
dis_optimizer = optim.Adam(dis.parameters(),lr)
SEQGAN = True
#pre_train_discriminator(dis, dis_optimizer, gen,10,train_iter) 


"""   
                    Uncomment to check outputs 


aise = Generator(SOS,EOU,SRC_VOCAB_SIZE,TGT_VOCAB_SIZE, GEN_HIDDEN_DIM, GEN_EMBEDDING_DIM, MAX_SEQ_LEN)
aise = aise.to(device)
asdd = torch.load("generator_checkpoint_pglrdiscount2.pth.tar")
aise.load_state_dict(asdd["state_dict"]) 
#aise.train_generator_MLE(genMLE_optimizer,train_iter,train_iter, MLE_TRAIN_EPOCHS,DEVICE)

print("You have done it")
from rouge import Rouge 

def CalcExs():
    asd = next(train_iter.__iter__())
    
    summary = asd.headlines
    test = asd.text
    sdf = torch.argmax(gen.forward(test,summary,0),dim=2)
    sdaf = torch.argmax(aise.forward(test,summary,0),dim=2)
    
    Text = []
    Original_Summary = []
    MLE = []
    GAN = []
    
    for j in range(summary.size(1)):
      txt = " "
      sas,d = test.size()
    #  print(sas)
      for i in range(sas):
        txt+=TEXT.vocab.itos[test[i][j]] + " "
      txt = txt.split("<eos>")[0].split("<sos>")[1]
      
      ss,d = summary.size()
      
      sa = " "
    
      for i in range(ss):
        sa+=SUMMARY.vocab.itos[summary[i][j]] + " "
      sa = sa.split("<eos>")[0].split("<sos>")[1]
      # print(sa)
    
      s = " "
      ss,sd =sdf.size()
      for i in range(ss):
        s+=SUMMARY.vocab.itos[sdf[i][j]] + " "
      s = s.split("<eos>")[0].split("<sos>")[1]
    
      ss,d = summary.size()
    
    
      sfs = " "
      for i in range(ss):
        sfs+=SUMMARY.vocab.itos[sdaf[i][j]] + " "
      sfs = sfs.split("<eos>")[0].split("<sos>")[1]
      
      Text.append(txt)
      Original_Summary.append(sa)
      MLE.append(s)
      GAN.append(sfs)
    
    diction = {
            "Text":Text,
            "Original Summary":Original_Summary,
            "MLE Summary":MLE,
            "GAN Summary":GAN
            }
    table = pd.DataFrame(diction)
    table.to_csv("StrongGenerator2.csv")
    
    tabula = pd.read_csv("StrongGenerator2.csv")
    print(tabula.head(10))
    
CalcExs() 
""" 



chkpts = torch.load("discriminator_checkpointlr8.pth.tar")
dis.load_state_dict(chkpts["state_dict"])
    

## Comment out below this to check outputs
########################################### ADVESRARIAL TRAINING ###############################################

print("Beginning Adversarial Training")


PG_optimizer = optim.Adagrad(gen.parameters(),lr=0.001)
num_policy_epochs = 10
for epoch in range(num_policy_epochs):
  counter = 0
  print("This is epoch:",epoch)
  for (i,result) in enumerate(train_iter):
    context = result.text.t()
    reply = result.headlines.t()
    #train_generator_PG(context.to(DEVICE), reply.to(DEVICE),gen,PG_optimizer,dis,num_samples=NUM_SAMPLES)
    TF=0
    fake_reply, word_probabilities, hiddens = gen.sample(context, reply, TF=TF)
    num_samples = NUM_SAMPLES
    if TF==1:
      if SEQGAN:
        rewards = torch.ones(BATCH_SIZE, MAX_SEQ_LEN-1).to(DEVICE)
      else:
        rewards = torch.ones(BATCH_SIZE, MAX_SEQ_LEN-1).to(DEVICE)
    # Compute word-level rewards
    elif SEQGAN:
      rewards = gen.monte_carlo(dis, context, fake_reply, hiddens, num_samples).detach()
    else:
      rewards, sentence_level_rewards = dis.get_rewards(fake_reply.long().to(DEVICE), PAD)
      entropy = torch.mean(word_probabilities, dim=1)
      perplexity = torch.mean(2**(-entropy)).item()
      
    

      # Compute REINFORCE loss with the assumption that G = R_t
    pg_loss = gen.compute_reinforce_loss(rewards.detach(), word_probabilities)
    if(counter%(10)==0):
      print(pg_loss,"sf")
      #print(i/len(train_iter),len(train_iter))
      torch.save({
        'epoch': epoch+1,
        'state_dict': gen.state_dict(),
        'optimizer' : PG_optimizer.state_dict()
       },'generator_checkpoint_pglrdiscount{}.pth.tar'.format(epoch))   
      
      

      # Backward pass
    PG_optimizer.zero_grad()
    pg_loss.backward()
    PG_optimizer.step()
    counter+=1



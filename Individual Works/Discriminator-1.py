#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 18:10:05 2019

@author: r17935avinash
"""

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
chkpts = torch.load("discriminator_checkpointlr8.pth.tar")
dis.load_state_dict(chkpts["state_dict"])


"""   
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

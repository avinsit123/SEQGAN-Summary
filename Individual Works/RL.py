#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 18:11:26 2019

@author: r17935avinash
"""

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

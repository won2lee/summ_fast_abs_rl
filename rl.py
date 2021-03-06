""" RL training utilities"""
import math
from time import time
from datetime import timedelta

from toolz.sandbox.core import unzip
from cytoolz import concat

import numpy as np
import torch
from torch.nn import functional as F
from torch import autograd
from torch.nn.utils import clip_grad_norm_

from metric import compute_rouge_l, compute_rouge_n
from training import BasicPipeline
from data.batcher import for_cnn

k_ceiling = 40
abs_ceiling = 5

def reverse_snts(snts):
    if snts[0][0] in ['_','^','`']:
        return [for_cnn(''.join(s)).split() for s in snts]
    else:
        return snts

def a2c_validate(agent, abstractor, loader, mono_abs):
    agent.eval()
    start = time()
    print('start running validation...', end='')
    avg_reward = 0
    i = 0
    max_k = min(k_ceiling,15)
    with torch.no_grad():
        for art_batch, abs_batch in loader:
            ext_sents = []
            ext_inds = []
            for raw_arts in art_batch:
                indices = agent(raw_arts)
                ext_inds += [(len(ext_sents), len(indices)-1)]
                extrctd = ([raw_arts[idx.item()]
                                for idx in indices if idx.item() < len(raw_arts)])
                if len(extrctd) < 1:
                    extrctd = ["_ 예정이 다 _ ."]
                if mono_abs==1:
                    ext_sent=[]
                    for ix,s in enumerate(extrctd): #[:3]:
                        if ix> max_k:
                            break
                        ext_sent+=s
                    ext_sents += [ext_sent] #[for s in extrctd[:3]]
                else:
                    ext_sents += extrctd

            #print(f"first ext_sents: {ext_sents[0]}")
            #print(f"last ext_sents: {ext_sents[-1]}")
            all_summs = reverse_snts(abstractor(ext_sents))
            #print(f"all_summs[0]: {all_summs[0]}")
            for ibatch, ((j, n), abs_sents) in enumerate(zip(ext_inds, abs_batch)):
                abs_sents = reverse_snts(abs_sents)
                #print(abs_sents)
                summs = [all_summs[ibatch]] if mono_abs==1 else all_summs[j:j+n]
                # python ROUGE-1 (not official evaluation)
                # print(f"abs_sents: {list(concat(abs_sents))}")
                # print(f"abs_sents: {list(concat([summs]))}")
                avg_reward += compute_rouge_n(list(concat(summs)),
                                              list(concat(abs_sents)), n=1)
                i += 1
                # if i%200 ==0:
                #     print(f"{i}th summ : {summs[0]}")
                #     print(f"{i}th abs sents: {abs_sents[0]}")
                #     print()

    avg_reward /= (i/100)
    print('finished in {}! avg reward: {:.2f}'.format(
        timedelta(seconds=int(time()-start)), avg_reward))
    return {'reward': avg_reward}


def a2c_train_step(agent, abstractor, loader, opt, grad_fn,
                   gamma=0.99, reward_fn=compute_rouge_l,
                   stop_reward_fn=compute_rouge_n(n=1), stop_coeff=1.0,
                   mono_abs=0, join_abs=False):
    opt.zero_grad()
    indices = []
    probs = []
    baselines = []
    ext_sents = []
    art_batch, abs_batch = next(loader)

    max_abs = abs_ceiling

    for raw_arts in art_batch:
        max_k = k_ceiling
        # if mono_abs:
        #     (inds, ms), bs = agent(raw_arts, n_abs=10000)
        # else:
        #     (inds, ms), bs = agent(raw_arts)
        (inds, ms), bs = agent(raw_arts)  
        inds = inds[:max_k]
        ms = ms[:max_k]
        bs = bs[:max_k]    

        if mono_abs==1:
            i_stop=1000
            for ix,idx in enumerate(inds):
                if idx.item() > len(raw_arts) -1:
                    i_stop=ix
                    break
            extrctd = [raw_arts[idx.item()]
                              for ix,idx in enumerate(inds) if idx.item() < len(raw_arts) and ix < i_stop ] 
            max_k = min(len(extrctd) + 1, max_k)
            # ext_sent = []
            # for i,ex in enumerate(extrctd):
            #     ext_sent += [ex]
            #     ext_sents +=[''.join(ext_sent)]

            # if ex is list, then as follows

            k = len(extrctd) #min(len(extrctd),3)
            ext_sent = [[] for _ in range(k)]
            #print(k,ext_sent)
            for i,ex in enumerate(extrctd):
                for j in range(i,len(extrctd)):
                    ext_sent[j] +=ex
                # if i<max_abs:
                #     for j in range(i,k):
                #         # ext_sent[j] +=ex
                #         ext_sent[j] +=ex #[ex]
                #     #ext_sent[i]=[' '.join(ext_sent[i])]
                # else:
                #     ext_sent[i] = "_ 예정이 다 _ ."
            ext_sents += [snts if ix < max_abs else "_ it _ is _ nothing ." for ix,snts in enumerate(ext_sent)]  

            inds = inds[:max_k]
            ms = ms[:max_k]
            bs = bs[:max_k]
         
        else:
            extrctd = [raw_arts[idx.item()]
                  for idx in inds if idx.item() < len(raw_arts)] # idc.item() >= len(raw_arts) ---> End of Extraction 
            ext_sents += extrctd

        baselines.append(bs)
        indices.append(inds)
        probs.append(ms)


    with torch.no_grad():
        summaries = reverse_snts(abstractor(ext_sents))
    #print(summaries[0])
    i = 0
    rewards = []
    avg_reward = 0
    for inds, abss in zip(indices, abs_batch):
        abss = reverse_snts(abss)
        #print(abss)
        # print(f"inds:{type(inds)}, abss:{type(abss)}")
        if mono_abs:
            #reward_fn = reward_fn(mode='r')
            abss = list(concat(abss))
            #print(f'i+j, summary.len : {i} , {min(len(inds), 3)},{len(summaries)}')
            cum_rwd = [0.]+[reward_fn(summaries[i+j] if mono_abs==1 else list(concat([summaries[jsub] for jsub in range(i,i+j)])), abss) #abss[0]) # cumulated rewards
                        for j in range(min(len(inds)-1, max_abs))]
            # if i%50==0:
            #     cumrwd = [f"{dr:.3f}" for dr in cum_rwd[:15]]
            #     print(f"cum_rwd : {cumrwd}") #" {avg_reward}")

            rs = ([max(cum_rwd[j+1]-cum_rwd[j], 0.0)   #contribution to total reward by one step action
                  for j in range(min(len(inds)-1, max_abs))]
                  + [0 for _ in range(max(0, len(inds)-1-max_abs))]
            #if len(rs) < 4:  # 3개 보다 많이 추출 했을 경우 stop_reward 를 주지 않은 방식 적용 
                  + [stop_coeff*stop_reward_fn(
                      list(concat([summaries[i+min(len(inds)-1, max_abs)-1]] if mono_abs==1 else [summaries[jsub] for jsub in range(i,i+min(len(inds)-1, max_abs)-1)])),
                      abss)]) # list(concat(abss)))])
        else:
            rs = ([reward_fn(summaries[i+j], abss[j] if not join_abs else list(concat(abss)))
                  for j in range(min(len(inds)-1, len(abss)))]
                  + [0 for _ in range(max(0, len(inds)-1-len(abss)))]
                  + [stop_coeff*stop_reward_fn(
                      list(concat(summaries[i:i+len(inds)-1])),
                      list(concat(abss)))])
        #print(f'rs:{len(rs)}, inds:{len(inds)}')
        #assert len(rs) == len(inds) +1 if mono_abs else len(inds)
        assert len(rs) == len(inds)
        avg_reward += rs[-1]/stop_coeff
        i += len(inds)-1
        # compute discounted rewards

        R = 0
        disc_rs = []
        for r in rs[::-1]:
            R = r + gamma * R
            disc_rs.insert(0, R)
        rewards += disc_rs
        if i%30==0:
            drs = [f"{dr:.2f}" for dr in disc_rs[:15]]
            print(f"rewards : {len(disc_rs)},  avg_rewards : {rs[-1]:.2f}, {drs}") #" {avg_reward}")
    indices = list(concat(indices))
    probs = list(concat(probs))
    baselines = list(concat(baselines))
    # standardize rewards
    reward = torch.Tensor(rewards).to(baselines[0].device)
    reward = (reward - reward.mean()) / (
        reward.std() + float(np.finfo(np.float32).eps))
    baseline = torch.cat(baselines).squeeze()
    avg_advantage = 0
    losses = []
    for action, p, r, b in zip(indices, probs, reward, baseline):
        advantage = r - b
        avg_advantage += advantage
        losses.append(-p.log_prob(action)
                      * (advantage/len(indices))) # divide by T*B
    #print(f"reward:{reward.size()}, baseline : {baseline.size()}")
    critic_loss = F.mse_loss(baseline, reward)
    #print(f"[critic_loss] + losses: ") #{[critic_loss] + losses}")
    #for mm in [critic_loss] + losses:
    #    print(mm)
    # backprop and update
    autograd.backward(
        [critic_loss] + losses
    )
    #, [torch.ones(1).to(critic_loss.device)]*(1+len(losses))
    grad_log = grad_fn()
    opt.step()
    log_dict = {}
    log_dict.update(grad_log)
    log_dict['reward'] = avg_reward/len(art_batch)
    log_dict['advantage'] = avg_advantage.item()/len(indices)
    log_dict['mse'] = critic_loss.item()
    assert not math.isnan(log_dict['grad_norm'])
    return log_dict


def get_grad_fn(agent, clip_grad, max_grad=1e2):
    """ monitor gradient for each sub-component"""
    params = [p for p in agent.parameters()]
    def f():
        grad_log = {}
        for n, m in agent.named_children():
            tot_grad = 0
            for p in m.parameters():
                if p.grad is not None:
                    tot_grad += p.grad.norm(2) ** 2
            tot_grad = tot_grad ** (1/2)
            grad_log['grad_norm'+n] = tot_grad.item()
        grad_norm = clip_grad_norm_(
            [p for p in params if p.requires_grad], clip_grad)
        grad_norm = grad_norm.item()
        if max_grad is not None and grad_norm >= max_grad:
            print('WARNING: Exploding Gradients {:.2f}'.format(grad_norm))
            grad_norm = max_grad
        grad_log['grad_norm'] = grad_norm
        return grad_log
    return f


class A2CPipeline(BasicPipeline):
    def __init__(self, name,
                 net, abstractor,
                 train_batcher, val_batcher,
                 optim, grad_fn,
                 reward_fn, gamma,
                 stop_reward_fn, stop_coeff,
                 mono_abs, join_abs):
        self.name = name
        self._net = net
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._opt = optim
        self._grad_fn = grad_fn

        self._abstractor = abstractor
        self._gamma = gamma
        self._reward_fn = reward_fn
        self._stop_reward_fn = stop_reward_fn
        self._stop_coeff = stop_coeff
        self._mono_abs = mono_abs
        self._join_abs = join_abs

        self._n_epoch = 0  # epoch not very useful?

    def batches(self):
        raise NotImplementedError('A2C does not use batcher')

    def train_step(self):
        # forward pass of model
        self._net.train()
        log_dict = a2c_train_step(
            self._net, self._abstractor,
            self._train_batcher,
            self._opt, self._grad_fn,
            self._gamma, self._reward_fn,
            self._stop_reward_fn, self._stop_coeff,
            mono_abs=self._mono_abs, join_abs=self._join_abs 
        )
        return log_dict

    def validate(self):
        return a2c_validate(self._net, self._abstractor, self._val_batcher, self._mono_abs)

    def checkpoint(self, *args, **kwargs):
        # explicitly use inherited function in case I forgot :)
        return super().checkpoint(*args, **kwargs)

    def terminate(self):
        pass  # No extra processs so do nothing

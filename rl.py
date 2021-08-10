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


def a2c_validate(agent, abstractor, loader, mono_abs):
    agent.eval()
    start = time()
    print('start running validation...', end='')
    avg_reward = 0
    i = 0
    with torch.no_grad():
        for art_batch, abs_batch in loader:
            ext_sents = []
            ext_inds = []
            for raw_arts in art_batch:
                indices = agent(raw_arts)
                ext_inds += [(len(ext_sents), len(indices)-1)]
                extrctd = ([raw_arts[idx.item()]
                                for idx in indices if idx.item() < len(raw_arts)])
                if mono_abs:
                    ext_sent=[]
                    for s in extrctd: #[:3]:
                        ext_sent+=s
                    ext_sents += [ext_sent] #[for s in extrctd[:3]]
                else:
                    ext_sents += extrctd

            #print(f"first ext_sents: {ext_sents[0]}")
            #print(f"last ext_sents: {ext_sents[-1]}")
            all_summs = abstractor(ext_sents)
            for ibatch, ((j, n), abs_sents) in enumerate(zip(ext_inds, abs_batch)):
                summs = [all_summs[ibatch]] if mono_abs else all_summs[j:j+n]
                # python ROUGE-1 (not official evaluation)
                # print(f"abs_sents: {list(concat(abs_sents))}")
                # print(f"abs_sents: {list(concat([summs]))}")
                avg_reward += compute_rouge_n(list(concat(summs)),
                                              list(concat(abs_sents)), n=1)
                i += 1
    avg_reward /= (i/100)
    print('finished in {}! avg reward: {:.2f}'.format(
        timedelta(seconds=int(time()-start)), avg_reward))
    return {'reward': avg_reward}


def a2c_train_step(agent, abstractor, loader, opt, grad_fn,
                   gamma=0.99, reward_fn=compute_rouge_l,
                   stop_reward_fn=compute_rouge_n(n=1), stop_coeff=1.0,
                   mono_abs=False):
    opt.zero_grad()
    indices = []
    probs = []
    baselines = []
    ext_sents = []
    art_batch, abs_batch = next(loader)

    max_abs = 4
    for raw_arts in art_batch:
        # if mono_abs:
        #     (inds, ms), bs = agent(raw_arts, n_abs=10000)
        # else:
        #     (inds, ms), bs = agent(raw_arts)
        (inds, ms), bs = agent(raw_arts)
        baselines.append(bs)
        indices.append(inds)
        probs.append(ms)

        extrctd = [raw_arts[idx.item()]
                          for idx in inds if idx.item() < len(raw_arts)] # idc.item() >= len(raw_arts) ---> End of Extraction 
        if mono_abs:
            # ext_sent = []
            # for i,ex in enumerate(extrctd):
            #     ext_sent += [ex]
            #     ext_sents +=[''.join(ext_sent)]

            # if ex is list, then as follows

            k = len(extrctd) #min(len(extrctd),3)
            ext_sent = [[] for _ in range(k)]
            #print(k,ext_sent)
            for i,ex in enumerate(extrctd):
                if i<max_abs:
                    for j in range(i,k):
                        # ext_sent[j] +=ex
                        ext_sent[j] +=ex #[ex]
                    #ext_sent[i]=[' '.join(ext_sent[i])]
                else:
                    ext_sent[i] = "_ 예정이 다 _ ."
            ext_sents += ext_sent           
         
        else:
            ext_sents += extrctd

    with torch.no_grad():
        summaries = abstractor(ext_sents)
    i = 0
    rewards = []
    avg_reward = 0
    for inds, abss in zip(indices, abs_batch):
        # print(f"inds:{type(inds)}, abss:{type(abss)}")
        if mono_abs:
            #print(f'i+j, summary.len : {i} , {min(len(inds), 3)},{len(summaries)}')
            cum_rwd = [0.]+[reward_fn(summaries[i+j], abss[0]) # cumulated rewards
                        for j in range(min(len(inds)-1, max_abs))]
            rs = ([cum_rwd[j+1]-cum_rwd[j]   #contribution to total reward by one step action
                  for j in range(min(len(inds)-1, max_abs))]
                  + [0 for _ in range(max(0, len(inds)-1-max_abs))]
            #if len(rs) < 4:  # 3개 보다 많이 추출 했을 경우 stop_reward 를 주지 않은 방식 적용 
                  + [stop_coeff*stop_reward_fn(
                      list(concat([summaries[i+min(len(inds)-1, max_abs)-1]])),
                      list(concat(abss)))])
        else:
            rs = ([reward_fn(summaries[i+j], abss[j])
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
                 mono_abs):
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
            mono_abs=self._mono_abs
        )
        return log_dict

    def validate(self):
        return a2c_validate(self._net, self._abstractor, self._val_batcher, self._mono_abs)

    def checkpoint(self, *args, **kwargs):
        # explicitly use inherited function in case I forgot :)
        return super().checkpoint(*args, **kwargs)

    def terminate(self):
        pass  # No extra processs so do nothing

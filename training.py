""" module providing basic training utilities"""
import os
from os.path import join, exists
from time import time
from datetime import timedelta
from itertools import starmap

from cytoolz import curry, reduce

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tensorboardX

seq_wgt = 1. #0.096 #0.09 
cov_wgt = 0.01 #0.1  #0.25

def get_basic_grad_fn(net, clip_grad, max_grad=1e2):
    def f():
        grad_norm = clip_grad_norm_(
            [p for p in net.parameters() if p.requires_grad], clip_grad)
        grad_norm = grad_norm.item()
        if max_grad is not None and grad_norm >= max_grad:
            print('WARNING: Exploding Gradients {:.2f}'.format(grad_norm))
            grad_norm = max_grad
        grad_log = {}
        grad_log['grad_norm'] = grad_norm
        return grad_log
    return f

@curry
def compute_loss(net, criterion, parallel,fw_args, loss_args):
    if parallel:
        logit, XO, cov_loss = net(*fw_args)
        loss1, mask = criterion(*((logit[0],) +  loss_args))
        #loss1, mask = criterion(*((logit[0],) +  loss_args))        
        loss2, _ = criterion(*((logit[1],) +  (XO,)), mask=mask)
        return (loss1, loss2, cov_loss)
    else:
        loss, _ = criterion(*((net(*fw_args)[0],) + loss_args))
        return loss

@curry
def val_step(loss_step, parallel, fw_args, loss_args):
    loss = loss_step(fw_args, loss_args)
    if parallel:
        return ((loss[0].size(0), loss[0].sum().item()), 
                (loss[1].size(0), loss[1].sum().item()), 
                (1, (sum(loss[2])/len(loss[2])).mean())
                ) 
    else:
        return loss.size(0), loss.sum().item()

@curry
def basic_validate(net, criterion, parallel, val_batches):
    print('running validation ... ', end='')
    net.eval()
    start = time()
    with torch.no_grad():
        validate_fn = val_step(compute_loss(net, criterion, parallel),parallel)
        #print(validate_fn(val_batches))
        if parallel:
            lgt1,lgt2,lgt3 = reduce(
                lambda a, b: [(a[i][0]+b[i][0], a[i][1]+b[i][1]) for i in range(3)],
                starmap(validate_fn, val_batches),
                [(0, 0),(0, 0),(0, 0)]
            )
        else:   
            n_data, tot_loss = reduce(
                lambda a, b: (a[0]+b[0], a[1]+b[1]),
                starmap(validate_fn, val_batches),
                (0, 0)
            )
    #print("got loss")
    val_loss = tot_loss / n_data if not parallel else seq_wgt * (lgt1[1]/lgt1[0] + lgt2[1]/lgt2[0]) + cov_wgt * lgt3[1]/lgt3[0]
    print(
        'validation finished in {}                                    '.format(
            timedelta(seconds=int(time()-start)))
    )
    print('validation loss: {:.4f} ... '.format(val_loss))
    if parallel:
        print(f" .... loss1 : {lgt1[1]/lgt1[0]}, loss2 : {lgt2[1]/lgt2[0]}, loss3 : {lgt3[1]/lgt3[0]}")
    return {'loss': val_loss}


class BasicPipeline(object):
    def __init__(self, name, net,
                 train_batcher, val_batcher, batch_size,
                 val_fn, criterion, optim, grad_fn=None, parallel=False):
        self.name = name
        self._net = net
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._criterion = criterion
        self._opt = optim
        # grad_fn is calleble without input args that modifyies gradient
        # it should return a dictionary of logging values
        self._grad_fn = grad_fn
        self._val_fn = val_fn

        self._n_epoch = 0  # epoch not very useful?
        self._batch_size = batch_size
        self._batches = self.batches()
        self.parallel = parallel
        self.count = 0

    def batches(self):
        while True:
            for fw_args, bw_args in self._train_batcher(self._batch_size):
                yield fw_args, bw_args
            self._n_epoch += 1

    def get_loss_args(self, net_out, bw_args):
        #print(F"net_out :{type(net_out)},  bw_args :{type(bw_args)}")
        if isinstance(net_out, tuple):
            loss_args = net_out + bw_args
        else:
            loss_args = (net_out, ) + bw_args
        return loss_args

    def train_step(self):
        # forward pass of model
        self.count+=1
        self._net.train()
        #print("start train_step")
        fw_args, bw_args = next(self._batches)
        #print("got one batch")
        #print(f"fw_args.size : {[fw_args[i].size() if type(fw_args[i]) is torch.Tensor else len(fw_args[i]) for i in range(4)]}")
        net_out, XO, cov_loss = self._net(*fw_args)
        if self.count%50==0 and self.parallel:
            print(f"XO[0]     : {XO[0][:20]}")
            print(f"inf XO[0] : {net_out[1][0][:20].argmax(-1)}")
            print(f"len(cov_loss) : {len(cov_loss)}") # max_abs 에서 XO 가 제거된 sequence 갯수 
        #print("one copy_summ process was done")

        # get logs and output for logging, backward
        log_dict = {}

        if self.parallel:
            #print("normal loss process")
            loss_args = self.get_loss_args(net_out[0], bw_args)
            loss1, mask = self._criterion(*loss_args)
            #loss = loss1.mean()
            #print("XO loss process")
            loss_args = self.get_loss_args(net_out[1], (XO,))
            loss2, _ = self._criterion(*loss_args, mask=mask)
            loss = seq_wgt * (loss1.mean() + loss2.mean()) + cov_wgt * (sum(cov_loss)/len(cov_loss)).mean()
            if self.count%50==0:
                print(f"loss of step {self.count} -- loss1 : {loss1.mean()}, loss2 : {loss2.mean()}, loss3 : {(sum(cov_loss)/len(cov_loss)).mean()}")
        else:
            loss_args = self.get_loss_args(net_out, bw_args)
            # backward and update ( and optional gradient monitoring )
            loss = self._criterion(*loss_args)[0].mean()
            if self.count%50==0:
                print(f"loss of step {self.count} : {loss.mean()}")

        loss.backward()
        #print(f"loss :{loss}")
        log_dict['loss'] = loss.item()
        if self._grad_fn is not None:
            log_dict.update(self._grad_fn())
        self._opt.step()
        self._net.zero_grad()

        return log_dict

    def validate(self):
        return self._val_fn(self._val_batcher(self._batch_size))

    def checkpoint(self, save_path, step, val_metric=None):
        save_dict = {}
        if val_metric is not None:
            name = 'ckpt-{:6f}-{}'.format(val_metric, step)
            save_dict['val_metric'] = val_metric
        else:
            name = 'ckpt-{}'.format(step)

        save_dict['state_dict'] = self._net.state_dict()
        save_dict['optimizer'] = self._opt.state_dict()
        torch.save(save_dict, join(save_path, name))

    def terminate(self):
        self._train_batcher.terminate()
        self._val_batcher.terminate()


class BasicTrainer(object):
    """ Basic trainer with minimal function and early stopping"""
    def __init__(self, pipeline, save_dir, ckpt_freq, patience,
                 scheduler=None, val_mode='loss'):
        assert isinstance(pipeline, BasicPipeline)
        assert val_mode in ['loss', 'score']
        self._pipeline = pipeline
        self._save_dir = save_dir
        self._logger = tensorboardX.SummaryWriter(join(save_dir, 'log'))
        if not exists(join(save_dir, 'ckpt')):
            os.makedirs(join(save_dir, 'ckpt'))

        self._ckpt_freq = ckpt_freq
        self._patience = patience
        self._sched = scheduler
        self._val_mode = val_mode

        self._step = 0
        self._running_loss = None
        # state vars for early stopping
        self._current_p = 0
        self._best_val = None

    def log(self, log_dict):
        loss = log_dict['loss'] if 'loss' in log_dict else log_dict['reward']
        if self._running_loss is not None:
            self._running_loss = 0.99*self._running_loss + 0.01*loss
        else:
            self._running_loss = loss
        print('train step: {}, {}: {:.4f}\r'.format(
            self._step,
            'loss' if 'loss' in log_dict else 'reward',
            self._running_loss), end='')
        for key, value in log_dict.items():
            self._logger.add_scalar(
                '{}_{}'.format(key, self._pipeline.name), value, self._step)

    def validate(self):
        print()
        print("before pipeline ####################################")
        val_log = self._pipeline.validate()
        print("after pipeline ####################################")
        for key, value in val_log.items():
            self._logger.add_scalar(
                'val_{}_{}'.format(key, self._pipeline.name),
                value, self._step
            )
        if 'reward' in val_log:
            val_metric = val_log['reward']
        else:
            val_metric = (val_log['loss'] if self._val_mode == 'loss'
                          else val_log['score'])
        return val_metric

    def checkpoint(self):
        val_metric = self.validate()
        self._pipeline.checkpoint(
            join(self._save_dir, 'ckpt'), self._step, val_metric)
        if isinstance(self._sched, ReduceLROnPlateau):
            self._sched.step(val_metric)
        else:
            self._sched.step()
        stop = self.check_stop(val_metric)
        return stop

    def check_stop(self, val_metric):
        if self._best_val is None:
            self._best_val = val_metric
        elif ((val_metric < self._best_val and self._val_mode == 'loss')
              or (val_metric > self._best_val and self._val_mode == 'score')):
            self._current_p = 0
            self._best_val = val_metric
        else:
            self._current_p += 1
        return self._current_p >= self._patience

    def train(self):
        try:
            start = time()
            print('Start training')
            while True:
                self._step += 1
                #print(f"{self._step} step was started")
                log_dict = self._pipeline.train_step()
                self.log(log_dict)

                if self._step % 50 == 0:
                    print(f"{self._step} step was done")

                if self._step % self._ckpt_freq == 0:
                    stop = self.checkpoint()
                    if stop:
                        break
            print('Training finised in ', timedelta(seconds=time()-start))
        finally:
            self._pipeline.terminate()

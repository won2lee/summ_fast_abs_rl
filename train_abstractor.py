""" train the abstractor"""
import argparse
import json
import os
from os.path import join, exists
import pickle as pkl

from cytoolz import compose

import torch
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from model.copy_summ import CopySumm
from model.util import sequence_loss
from training import get_basic_grad_fn, basic_validate
from training import BasicPipeline, BasicTrainer
from decoding import load_best_ckpt

from data.data import CnnDmDataset
from data.batcher import coll_fn, prepro_fn
from data.batcher import convert_batch_copy, batchify_fn_copy
from data.batcher import BucketedGenerater

from utils import PAD, UNK, START, END
from utils import (make_vocab, make_embedding, 
                   make_embedding_from_pretrained, 
                   apply_sub_module_weight_from_pretrained)

# NOTE: bucket size too large may sacrifice randomness,
#       to low may increase # of PAD tokens
BUCKET_SIZE = 6400 #00

try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')

class MatchDataset(CnnDmDataset):
    """ single article sentence -> single abstract sentence
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split, mono_abs):
        super().__init__(split, DATA_DIR, mono_abs)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, abs_sents, extracts = (
            js_data['article'], js_data['abstract'], js_data['extracted'])
        matched_arts = [art_sents[i] for i in extracts]
        if self._mono_abs:
            matched_arts = [" ".join(matched_arts)]
        return matched_arts, abs_sents[:1 if self._mono_abs else len(extracts)]


def configure_net(vocab_size, emb_dim,
                  n_hidden, bidirectional, n_layer, parallel, use_coverage):
    net_args = {}
    net_args['vocab_size']    = vocab_size
    net_args['emb_dim']       = emb_dim
    net_args['n_hidden']      = n_hidden
    net_args['bidirectional'] = bidirectional
    net_args['n_layer']       = n_layer
    net_args['parallel']      = parallel
    net_args['use_coverage']  = use_coverage

    net = CopySumm(**net_args)
    return net, net_args


def configure_training(opt, lr, clip_grad, lr_decay, batch_size):
    """ supports Adam optimizer only"""
    assert opt in ['adam']
    opt_kwargs = {}
    opt_kwargs['lr'] = lr

    train_params = {}
    train_params['optimizer']      = (opt, opt_kwargs)
    train_params['clip_grad_norm'] = clip_grad
    train_params['batch_size']     = batch_size
    train_params['lr_decay']       = lr_decay

    nll = lambda logit, target: F.nll_loss(logit, target, reduce=False)
    def criterion(logits, targets, mask=None):
        return sequence_loss(logits, targets, nll, pad_idx=PAD, mask=mask)

    return criterion, train_params

def build_batchers(word2id, cuda, debug, parallel, mono_abs):
    prepro = prepro_fn(args.max_art, args.max_abs, parallel=parallel)
    def sort_key(sample):
        src, target = sample
        return (len(target), len(src))
    batchify = compose(
        batchify_fn_copy(PAD, START, END, cuda=cuda, parallel=parallel),
        convert_batch_copy(UNK, word2id)
    )

    train_loader = DataLoader(
        MatchDataset('train', mono_abs), batch_size=BUCKET_SIZE,
        shuffle=not debug,
        num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn
    )
    train_batcher = BucketedGenerater(train_loader, prepro, sort_key, batchify,
                                      single_run=args.mono_abs, fork=not debug)

    val_loader = DataLoader(
        MatchDataset('val', mono_abs), batch_size=BUCKET_SIZE,
        shuffle=False, num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn
    )
    val_batcher = BucketedGenerater(val_loader, prepro, sort_key, batchify,
                                    single_run=args.mono_abs, fork=not debug)
    return train_batcher, val_batcher

def main(args):
    # create data batcher, vocabulary
    # batcher
    with open(join(DATA_DIR, 'vocab_cnt.pkl'), 'rb') as f:
        wc = pkl.load(f, encoding="bytes") 
    parallel = args.parallel
    print(f"main.parallel : {parallel}")
    word2id = make_vocab(wc, args.vsize)
    train_batcher, val_batcher = build_batchers(word2id,
                                                args.cuda, args.debug, parallel, args.mono_abs)

    # make net
    net, net_args = configure_net(len(word2id), args.emb_dim,
                                  args.n_hidden, args.bi, args.n_layer, parallel, args.use_coverage)
    
    if args.continued:
        # abs_meta = json.load(open(join(args.path, 'meta.json')))
        # assert abs_meta['net'] == 'base_abstractor'
        # abs_args = abs_meta['net_args']
        abs_ckpt = load_best_ckpt(args.path)
        # word2id = pkl.load(open(join(args.path, 'vocab.pkl'), 'rb'))
        net.load_state_dict(abs_ckpt)
        if args.lr < 0.00001:  # temporary
            net.requires_grad_(requires_grad=True)
        # self._device = torch.device('cuda' if cuda else 'cpu')
        # self._net = abstractor.to(self._device)
        # self._word2id = word2id
        # self._id2word = {i: w for w, i in word2id.items()}
        # self._max_len = max_len
        # self.parallel = abs_args['parallel']
        elif parallel and args.pretrained:
            pre_trained = torch.load(args.pretrained)
            net = apply_sub_module_weight_from_pretrained(
                    net,pre_trained, 
                    no_grad = True if args.lr > 0.0001 else False
                    )

    elif args.w2v or args.pretrained:
        if args.w2v:
            # NOTE: the pretrained embedding having the same dimension
            #       as args.emb_dim should already be trained
            embedding, _ = make_embedding(
                {i: w for w, i in word2id.items()}, args.w2v)
        else:
            pre_trained = torch.load(args.pretrained)
            embedding, _ = make_embedding_from_pretrained(
                {i: w for w, i in word2id.items()}, pre_trained)
            if parallel:
                net = apply_sub_module_weight_from_pretrained(
                        net,pre_trained,lang = args.lang,
                        no_grad = True
                        )

        net.set_embedding(embedding)

    # configure training setting
    criterion, train_params = configure_training(
        'adam', args.lr, args.clip, args.decay, args.batch
    )

    # save experiment setting
    if not exists(args.path):
        os.makedirs(args.path)
    with open(join(args.path, 'vocab.pkl'), 'wb') as f:
        pkl.dump(word2id, f, pkl.HIGHEST_PROTOCOL)
    meta = {}
    meta['net']           = 'base_abstractor'
    meta['net_args']      = net_args
    meta['traing_params'] = train_params
    with open(join(args.path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)

    # prepare trainer
    val_fn = basic_validate(net, criterion, parallel)
    grad_fn = get_basic_grad_fn(net, args.clip)
    optimizer = optim.Adam(net.parameters(), **train_params['optimizer'][1])
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                  factor=args.decay, min_lr=0,
                                  patience=args.lr_p)

    if args.cuda:
        net = net.cuda()
    pipeline = BasicPipeline(meta['net'], net,
                             train_batcher, val_batcher, args.batch, val_fn,
                             criterion, optimizer, grad_fn, parallel)
    trainer = BasicTrainer(pipeline, args.path,
                           args.ckpt_freq, args.patience, scheduler)

    print('start training with the following hyper-parameters:')
    print(meta)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training of the abstractor (ML)'
    )
    parser.add_argument('--path', required=True, help='root of the model')


    parser.add_argument('--vsize', type=int, action='store', default=30000,
                        help='vocabulary size')
    parser.add_argument('--emb_dim', type=int, action='store', default=128,
                        help='the dimension of word embedding')
    parser.add_argument('--w2v', action='store',
                        help='use pretrained word2vec embedding')
    parser.add_argument('--n_hidden', type=int, action='store', default=256,
                        help='the number of hidden units of LSTM')
    parser.add_argument('--n_layer', type=int, action='store', default=1,
                        help='the number of layers of LSTM')
    parser.add_argument('--no-bi', action='store_true',
                        help='disable bidirectional LSTM encoder')

    # length limit
    parser.add_argument('--max_art', type=int, action='store', default=100,
                        help='maximun words in a single article sentence')
    parser.add_argument('--max_abs', type=int, action='store', default=60,
                        help='maximun words in a single abstract sentence')
    # training options
    parser.add_argument('--lr', type=float, action='store', default=1e-3,
                        help='learning rate')
    parser.add_argument('--decay', type=float, action='store', default=0.5,
                        help='learning rate decay ratio')
    parser.add_argument('--lr_p', type=int, action='store', default=1,
                        help='patience for learning rate decay')
    parser.add_argument('--clip', type=float, action='store', default=2.0,
                        help='gradient clipping')
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='the training batch size')
    parser.add_argument(
        '--ckpt_freq', type=int, action='store', default=3000,
        help='number of update steps for checkpoint and validation'
    )
    parser.add_argument('--patience', type=int, action='store', default=5,
                        help='patience for early stopping')

    parser.add_argument('--debug', action='store_true',
                        help='run in debugging mode')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    parser.add_argument('--parallel', action='store_true',
                        help='enable sub lstm')
    parser.add_argument('--pretrained', action='store',
                        help='use pretrained-in-nmt embed')
    parser.add_argument('--continued', action='store_true',
                        help='use pretrained-abstrator')
    parser.add_argument('--mono_abs', action='store_true',
                        help='for kor summ data')
    parser.add_argument('--use_coverage', action='store_true',
                        help='for coverage')
    parser.add_argument('--lang', type=str, action='store', default='en',
                        help='language to summ')

    args = parser.parse_args()
    args.bi = not args.no_bi
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    main(args)

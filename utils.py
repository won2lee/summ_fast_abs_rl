""" utility functions"""
import re
import os
from os.path import basename

import gensim
import torch
from torch import nn
import copy


def count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


PAD = 0
UNK = 1
START = 2
END = 3

SPC = 4 #space
TTL = 5 #title
CAP = 6 #capital

def make_vocab(wc, vocab_size):
    word2id, id2word = {}, {}
    word2id['<pad>'] = PAD
    word2id['<unk>'] = UNK
    word2id['<start>'] = START
    word2id['<end>'] = END

    word2id['_'] = SPC
    word2id['^'] = TTL
    word2id['`'] = CAP

    i = 7
    for (w, _) in wc.most_common(vocab_size):
        if w in ['_','^','`']:
            continue
    # for i, (w, _) in enumerate(wc.most_common(vocab_size), 7):
        word2id[w] = i
        i += 1
    return word2id


def make_embedding(id2word, w2v_file, initializer=None):
    attrs = basename(w2v_file).split('.')  #word2vec.{dim}d.{vsize}k.bin
    w2v = gensim.models.Word2Vec.load(w2v_file).wv
    vocab_size = len(id2word)
    emb_dim = int(attrs[-3][:-1])
    embedding = nn.Embedding(vocab_size, emb_dim).weight
    if initializer is not None:
        initializer(embedding)

    oovs = []
    with torch.no_grad():
        for i in range(len(id2word)):
            # NOTE: id2word can be list or dict
            if i == START:
                embedding[i, :] = torch.Tensor(w2v['<s>'])
            elif i == END:
                embedding[i, :] = torch.Tensor(w2v[r'<\s>'])
            elif id2word[i] in w2v:
                embedding[i, :] = torch.Tensor(w2v[id2word[i]])
            else:
                oovs.append(i)
    return embedding, oovs

def make_embedding_from_pretrained(id2word, pre_trained, initializer=None):
    #attrs = basename(w2v_file).split('.')  #word2vec.{dim}d.{vsize}k.bin
    
    w2v = pre_trained['voc']
    vocab_size = len(id2word)
    #print(f"pre_keys : {pre_trained.keys()}")
    voc_embed = pre_trained['emb']['model_embeddings.vocabs.weight']
    emb_dim = voc_embed.size(-1)
    embedding = nn.Embedding(vocab_size, emb_dim).weight
    if initializer is not None:
        initializer(embedding)

    oovs = []
    with torch.no_grad():
        for i in range(len(id2word)):
            # NOTE: id2word can be list or dict
            if i == START:
                embedding[i, :] = torch.Tensor(voc_embed[w2v['<s>']])
            elif i == END:
                embedding[i, :] = torch.Tensor(voc_embed[w2v[r'</s>']])
            elif i in [4,5,6]:
                continue
            elif id2word[i] in w2v:
                embedding[i, :] = torch.Tensor(voc_embed[w2v[id2word[i]]])
            else:
                oovs.append(i)
    return embedding, oovs 

def apply_sub_module_weight_from_pretrained(net, pre_trained, lang='en', extr=False, no_grad=False):
    import re
    #keys() = {'en':pre_trained['en'].keys(), 'ko':pre_trained['ko'].keys()}
    p = re.compile("_(en|ko)_")
    q = re.compile("^(?!(sub)).*?_")

    m_keys ={}
    m_keys = {q.sub('sub_',p.sub('_',k)):k for k in pre_trained[lang].keys()}
    print(f"m_keys:{m_keys}")

    with torch.no_grad() if no_grad else torch.enable_grad():
        for k,v in m_keys.items():
            globals()['net._sent_enc.'+k if extr else 'net.'+k] = pre_trained[lang][v]
    return net

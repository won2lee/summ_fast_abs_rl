from nmt_model import NMT
import torch
from vocab import Vocab, get_wid2cid

params = torch.load('model.bin', map_location=lambda storage, loc: storage)
args = params['args']
wid2cid = get_wid2cid()
model = NMT(vocab=params['vocab'], char_size=85, wid2cid=wid2cid, **args)
model.load_state_dict(params['state_dict'])

#torch.save(params['state_dict'], 'model_bi_1114')
import json
vocs = json.load(open("vocab.json"))["vocs_word2id"]


keys={'voc': None,
 'emb': ['model_embeddings.vocabs.weight'],
 'en': ['sub_en_coder.weight_ih_l0',
  'sub_en_coder.weight_hh_l0',
  'sub_en_coder.bias_ih_l0',
  'sub_en_coder.bias_hh_l0',
  'sub_en_projection.weight',
  'en_gate.weight'],
 'ko': ['sub_ko_coder.weight_ih_l0',
  'sub_ko_coder.weight_hh_l0',
  'sub_ko_coder.bias_ih_l0',
  'sub_ko_coder.bias_hh_l0',
  'sub_ko_projection.weight',
  'ko_gate.weight']}

import re

lang = ["en","ko"]
lang_keys = {}
p = re.compile("_(en|ko)_")
q = re.compile("^(?!(sub)).*?_")
for l in lang:
    lang_keys[l] ={}
    lang_keys[l] = {q.sub('sub_',p.sub('_',k)):k for k in keys[l]}

pre_tr = {}
for k in list(keys.keys()):
    if k == 'voc':
        pre_tr[k] = vocs
    else:
        pre_tr[k]={}
        for k_sub in keys[k]:
            pre_tr[k][k_sub]=params['state_dict'][k_sub]
            
torch.save(pre_tr,"pre_trained")

aa = torch.load("pre_trained")
aa['ko']["sub_ko_coder.weight_ih_l0"].size()
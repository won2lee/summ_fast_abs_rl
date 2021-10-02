""" run decoding of rnn-ext + abs + RL (+ rerank)"""
import argparse
import json
import os
from os.path import join
from datetime import timedelta
from time import time
from collections import Counter, defaultdict
from itertools import product, chain
from functools import reduce
import operator as op

from cytoolz import identity, concat, curry

import torch
from torch.utils.data import DataLoader
from torch import multiprocessing as mp

from data.batcher import tokenize

from decoding import Abstractor, RLExtractor, DecodeDataset, BeamAbstractor
from decoding import make_html_safe


def decode(save_path, model_dir, split, batch_size,
           beam_size, diverse, max_len, cuda, mono_abs):
    start = time()
    # setup model
    with open(join(model_dir, 'meta.json')) as f:
        meta = json.loads(f.read())
    if meta['net_args']['abstractor'] is None:
        # NOTE: if no abstractor is provided then
        #       the whole model would be extractive summarization
        assert beam_size == 1
        abstractor = identity
    else:
        if beam_size == 1:  
            abstractor = Abstractor(join(model_dir, 'abstractor'),
                                    max_len, cuda)
        else:
            abstractor = BeamAbstractor(join(model_dir, 'abstractor'),
                                        max_len, cuda)
    extractor = RLExtractor(model_dir, cuda=cuda)

    # setup loader
    def coll(batch):
        articles = list(filter(bool, batch))
        return articles
    dataset = DecodeDataset(split, mono_abs)  # mono_abs is not used

    n_data = len(dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=coll
    )

    # prepare save paths and logs
    os.makedirs(join(save_path, 'output'))
    os.makedirs(join(save_path, 'in_out'))
    dec_log = {}
    dec_log['abstractor'] = meta['net_args']['abstractor']
    dec_log['extractor'] = meta['net_args']['extractor']
    dec_log['rl'] = True
    dec_log['split'] = split
    dec_log['beam'] = beam_size
    dec_log['diverse'] = diverse
    with open(join(save_path, 'log.json'), 'w') as f:
        json.dump(dec_log, f, indent=4)

    # Decoding
    i = 0
    with torch.no_grad():
        sb = {1:"_",2:"^",3:"`"}
        for i_debug, raw_article_batch in enumerate(loader):
            print(f"i_debug : {i_debug}")
            raw_articles= [bt["article"] for bt in raw_article_batch]
            #raw_ext = [bt["extracted"] for bt in raw_article_batch]
            raw_abs = [bt["abstract"] for bt in raw_article_batch]

            tokenized_article_batch = map(tokenize(None), raw_articles)
            #tokenized_article_batch = map(tokenize(None), raw_article_batch)
            raw_arts = []
            ext_arts = []
            ext_inds = []
            extrctd = []
            for raw_art_sents in tokenized_article_batch:
                raw_arts.append(raw_art_sents)
                ext = extractor(raw_art_sents)[:-1]  # exclude EOE
                if not ext:
                    # use top-5 if nothing is extracted
                    # in some rare cases rnn-ext does not extract at all
                    ext = list(range(5))[:len(raw_art_sents)]
                else:
                    ext = [i.item() for i in ext]
                ext_inds += [(len(ext_arts), len(ext))]
                extrctd.append(ext)
                if mono_abs:
                    ext_arts.append(list(chain(*[raw_art_sents[i] for i in ext])))
                else:
                    ext_arts += [raw_art_sents[i] for i in ext]

            print('extracted was done')
            if beam_size > 1:
                all_beams = abstractor(ext_arts, beam_size, diverse)
                dec_outs = rerank_mp(all_beams, ext_inds, mono_abs)
            else:
                dec_outs = abstractor(ext_arts)
            assert i == batch_size*i_debug

            print('abstracted was done')
            for ibt, (j, n) in enumerate(ext_inds):
                # decoded_sents = [' '.join(dec) for dec in dec_outs[j:j+n]]
                # decoded_sents = ([' '.join(list(chain(*[[w] if dec[1][i] == 0 else [sb[dec[1][i]], w] 
                #                 for i,w in enumerate(dec[0])])))
                #                 for dec in dec_outs[j:j+n]])  #in zip(dec_outs[0][j:j+n],dec_outs[1][j:j+n])])))])
                #print(f"dec_outs : {dec_outs[j:j+n]}")
                if beam_size > 1:
                    if mono_abs:
                        xs = dec_outs[ibt][1][1:] # why [1:] ==>_process_beam 에서 첫번째 sequence(start) 제거  
                        decoded_sents = ([''.join(list(chain(*[[str(w)] if xs[iw] == 0 else [sb[xs[iw]], str(w)] 
                                        for iw,w in enumerate(dec_outs[ibt][0])])))])
                    else:                     
                        # decoded_sents = ([''.join(list(chain(*[[str(w)] if xs[iw+1] == 0 else [sb[xs[iw+1]], str(w)] 
                        #                 for iw,w in enumerate(snt)])))
                        #                 for snt,xs in dec_outs[j:j+n]])
                        decoded_sents = ([' '.join(snt)
                                        for snt,xs in dec_outs[j:j+n]])                        
                else:
                    if mono_abs:
                        decoded_sents = ([' '.join(dec_outs[ibt])])
                    else: 
                        decoded_sents = ([' '.join(snt)
                                        for snt in dec_outs[j:j+n]])
                with open(join(save_path, 'output/{}.dec'.format(i)),
                          'w') as f:
                    f.write(make_html_safe('\n'.join(decoded_sents)))

                ######           added to check output's relevance          ##########
                if mono_abs:
                    in_out = {}
                    in_out["raw_arts"] = [''.join(snt) for snt in raw_arts[ibt]]
                    in_out['raw_exts'] = [in_out["raw_arts"][idx] for idx in raw_ext[ibt]]
                    in_out["raw_abss"] = [''.join(snt.split()) for snt in raw_abs[ibt]]
                    in_out["extracted"] = [in_out["raw_arts"][idx] for idx in extrctd[ibt]]
                    in_out["abstract"] = decoded_sents
                    
                    with open(join(save_path, 'in_out/{}.json'.format(i)),
                              'w') as jsonf:
                        json.dump(in_out,jsonf, ensure_ascii=False, indent=4)
                ######################################################################

                i += 1
                if i%10 == 0:
                    print('{}/{} ({:.2f}%) decoded in {} seconds'.format(  #\r'.format(
                        i, n_data, i/n_data*100,
                        timedelta(seconds=int(time()-start))
                    )) #, end='')
            print(f"raw_art_sents : {raw_art_sents}")
            print(f"ext_arts : {ext_arts[-1] if mono_abs else ext_arts[-len(ext):]}")
            print(f"decoded_sents : {decoded_sents}")

                    
    print()
    print("decoding was completed !!")

_PRUNE = defaultdict(
    lambda: 2,
    {1:5, 2:5, 3:5, 4:5, 5:5, 6:4, 7:3, 8:3}
)            # key : num of extracted sents, value: num of beam to compute => to decrease computation burden of product (key * beam) 
def rerank(all_beams, ext_inds):
    beam_lists = (all_beams[i: i+n] for i, n in ext_inds if n > 0)
    return list(concat(map(rerank_one, beam_lists)))

def rerank_mp(all_beams, ext_inds, mono_abs):
    if mono_abs:
        beam_lists = [[beams] for beams in all_beams]
    else:
        beam_lists = [all_beams[i: i+n] for i, n in ext_inds if n > 0]
    # with mp.Pool(8) as pool:
    #     reranked = pool.map(rerank_one, beam_lists)
    reranked = map(rerank_one, beam_lists)
    return list(concat(reranked))

def rerank_one(beams):
    @curry
    def process_beam(beam, n):
        for b in beam[:n]:
            b.gram_cnt = Counter(_make_n_gram(b.sequence))
        return beam[:n]
    beams = map(process_beam(n=_PRUNE[len(beams)]), beams)  # for example, if len(beam) == 7(extrctd snts ==7) 
                                                            #       then consider only 3 of 5 beams
    best_hyps = max(product(*beams), key=_compute_score)    #       i.e., prune 7 * 5 to 7 * 3  
    dec_outs = [(h.sequence, h.xo) for h in best_hyps]
    return dec_outs

def _make_n_gram(sequence, n=3):  # n : 2 => 3 for parallel 
    return (tuple(sequence[i:i+n]) for i in range(len(sequence)-(n-1)))

def _compute_score(hyps):
    all_cnt = reduce(op.iadd, (h.gram_cnt for h in hyps), Counter())
    repeat = sum(c-1 for g, c in all_cnt.items() if c > 1)
    lp = sum(h.logprob for h in hyps) / sum(len(h.sequence) for h in hyps)
    return -(repeat+3)**3.0 * (1/lp)  #(-repeat, lp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='run decoding of the full model (RL)')
    parser.add_argument('--path', required=True, help='path to store/eval')
    parser.add_argument('--model_dir', help='root of the full model')

    # dataset split
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument('--val', action='store_true', help='use validation set')
    data.add_argument('--test', action='store_true', help='use test set')

    # decode options
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='batch size of faster decoding')
    parser.add_argument('--beam', type=int, action='store', default=1,
                        help='beam size for beam-search (reranking included)')
    parser.add_argument('--div', type=float, action='store', default=1.0,
                        help='diverse ratio for the diverse beam-search')
    parser.add_argument('--max_dec_word', type=int, action='store', default=30,
                        help='maximun words to be decoded for the abstractor')

    parser.add_argument('--no_cuda', action='store_true',
                        help='disable GPU training')
    parser.add_argument('--mono_abs', action='store_true',
                        help='for kor summ data')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    data_split = 'test' if args.test else 'val'
    decode(args.path, args.model_dir,
           data_split, args.batch, args.beam, args.div,
           args.max_dec_word, args.cuda, args.mono_abs)

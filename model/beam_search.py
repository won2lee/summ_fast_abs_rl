""" beam-search utilities"""
from collections import Counter

from cytoolz import concat

import torch


class _Hypothesis(object):
    def __init__(self, sequence, logprob, hists, xo, init_vecs, attns=[]):
        """
        seqence: list of int tokens
        logprob: current log probability
        hists: history of prevous convolution list(n_layers)/
               prev_states and output of lstm ((H, C), out)
        """
        self.sequence = sequence
        self.logprob = logprob
        self.hists = hists
        self.attns = attns  # for unk replacement
        self.xo = xo
        self.init_vecs = init_vecs

    def extend_k(self, topk, logprobs, hists, attn=None, xok=None, init_vecs=None, diverse=1.0):
        if attn is None:
            attns = []
        else:
            attns = self.attns + [attn]
        print(f"topk :{topk}")
        print(f"logprobs : {logprobs}")
        print(f"xok : {xok}")
        return [_Hypothesis(self.sequence+[t.item()],
                            self.logprob+lp.item()-diverse*i, hists, 
                            self.xo+[x.item()], init_vecs, attns)
                for i, (t, lp, x) in enumerate(zip(topk, logprobs, xok))]

    def __lt__(self, other):
        return (other.logprob/len(other.sequence)
                < self.logprob/len(self.sequence))


def init_beam(start, hists, xo, init_vecs):
    """ get a initial beam to start beam search"""
    return [_Hypothesis([start], 0, hists, xo, init_vecs)]


def create_beam(tok, lp, hists):
    """ initailiza a beam with top k token"""
    k = tok.size(0)
    return [_Hypothesis([tok[i].item()], lp[i].item(), hists)
            for i in range(k)]


def pack_beam(hyps, device):
    """pack a list of hypothesis to decoder input batches"""
    token = torch.LongTensor([h.sequence[-1] for h in hyps])

    hists = tuple(torch.stack([hyp.hists[i] for hyp in hyps], dim=d)
                  for i, d in enumerate([1, 1, 0]))
    token = token.to(device)
    states = ((hists[0], hists[1]), hists[2])

    if hyps[0].init_vecs:
        xos = torch.LongTensor([h.xo[-1] for h in hyps])
        xos = xos.to(device)
        init_vecs = tuple(torch.stack([hyp.init_vecs[i] for hyp in hyps], dim=1) for i in range(2))
        return token, states, xos, init_vecs

    else:
        return token, states, None, None


def next_search_beam(beam, beam_size, finished,
                     end, topk, lp, hists, attn=None, xok=None, sub_stts=None, diverse=1.0):
    """generate the next beam(K-best hyps)"""
    topks, lps, hists_list, attns, xoks, sub_list = _unpack_topk(topk, lp, hists, attn, xok, sub_stts)
    hyps_lists = [h.extend_k(topks[i], lps[i],
                             hists_list[i], attns[i], xoks[i], sub_list[i], diverse)
                  for i, h in enumerate(beam)]
    hyps = list(concat(hyps_lists))
    finished, beam = _clean_beam(finished, hyps, end, beam_size)

    return finished, beam


def best_sequence(finished, beam=None):
    """ return the sequence with the highest prob(normalized by length)"""
    if beam is None:  # not empty
        best_beam = finished[0]
    else:
        if finished and beam[0] < finished[0]:
            best_beam = finished[0]
        else:
            best_beam = beam[0]

    best_seq = best_beam.sequence[1:]
    if best_beam.attns:
        return best_seq, best_beam.attns
    else:
        return best_seq


def _unpack_topk(topk, lp, hists, attn=None, xok=None, sub_stts=None):
    """unpack the decoder output"""
    beam, _ = topk.size()
    topks = [t for t in topk]
    lps = [l for l in lp]
    k_hists = [(hists[0][:, i, :], hists[1][:, i, :], hists[2][i, :])
               for i in range(beam)]
    if xok is not None:
        xoks = [x for x in xok]
        k_subs = [(sub_stts[0][:,i,:], sub_stts[1][:,i,:]) for i in range(beam)]
    else:
        xoks, k_subs = None,None 

    if attn is None:
        return topks, lps, k_hists
    else:
        attns = [attn[i] for i in range(beam)]
        return topks, lps, k_hists, attns, xoks, k_subs


def _clean_beam(finished, beam, end_tok, beam_size, remove_tri=True):
    """ remove completed sequence from beam """
    new_beam = []
    for h in sorted(beam, reverse=True,
                    key=lambda h: h.logprob/len(h.sequence)):
        if remove_tri and _has_repeat_tri(h.sequence):
            h.logprob = -1e9
        if h.sequence[-1] == end_tok:
            finished_hyp = _Hypothesis(h.sequence[:-1], # remove EOS
                                       h.logprob, h.hists, h.attns)
            finished.append(finished_hyp)
        else:
            new_beam.append(h)
        if len(new_beam) == beam_size:
            break
    else:
        # ensure beam size
        while len(new_beam) < beam_size:
            new_beam.append(new_beam[0])

    finished = sorted(finished, reverse=True,
                      key=lambda h: h.logprob/len(h.sequence))
    return finished, new_beam


def _has_repeat_tri(grams):
    tri_grams = [tuple(grams[i:i+3]) for i in range(len(grams)-2)]
    cnt = Counter(tri_grams)
    return not all((cnt[g] <= 1 for g in cnt))

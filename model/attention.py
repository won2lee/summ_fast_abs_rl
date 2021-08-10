""" attention functions """
import torch
from torch.nn import functional as F


def dot_attention_score(key, query):
    """[B, Tk, D], [(Bs), B, Tq, D] -> [(Bs), B, Tq, Tk]"""
    return query.matmul(key.transpose(1, 2))

def prob_normalize(score, mask):
    """ [(...), T]
    user should handle mask shape"""
    score = score.masked_fill(mask == 0, -1e18)
    norm_score = F.softmax(score, dim=-1)
    return norm_score

def attention_aggregate(value, score):
    """[B, Tv, D], [(Bs), B, Tq, Tv] -> [(Bs), B, Tq, D]"""
    output = score.matmul(value)
    return output


def coverage_score(key, query, cov, to_avoid):
    vT, enc_proj, dec_proj, w_c = cov
    # if type(to_avoid) is torch.Tensor:
    #     print(f'to_avoid : {to_avoid.size()}')
    # print(f'key :{key.size()},  query : {query.size()}')
    attn_proj = enc_proj(key) + dec_proj(query.unsqueeze(-2))
    # print(f"attn_proj : {attn_proj.size()}")

    if type(to_avoid) is torch.Tensor:
        attn_proj += w_c(to_avoid.unsqueeze(-1))
    # score = vT(F.tanh(attn_proj))
    # # print(f"score : {score.size()}")
    # score = vT(score)
    # # print(f"score : {score.size()}")
    return vT(F.tanh(attn_proj)).transpose(1,2)

def step_attention(query, key, value, mem_mask=None, cov=None, to_avoid=None):
    """ query[(Bs), B, D], key[B, T, D], value[B, T, D]"""
    #print(f"key :{key.size()}, query :{query.size()}")
    #print(f'cov ============================== {cov}')
    if cov is not None:
        score = coverage_score(key, query, cov, to_avoid)
    else:
        score = dot_attention_score(key, query.unsqueeze(-2))
    # print(f"score :{type(score)}")
    # print(f"score :{score.size()}")
    # if type(to_avoid) is torch.Tensor:
    #     print(f"score :{score.size()}, to_void : {to_avoid.size()},mem_mask :{mem_mask.size()}")      
    # if type(to_avoid) is torch.Tensor:
    #     score = cov(torch.cat((score,to_avoid.unsqueeze(1).expand_as(score)), -2).transpose(1,2)).transpose(1,2)

    if mem_mask is None:
        norm_score = F.softmax(score, dim=-1)
    else:
        # if mem_mask.size()[-1] > score.size()[-1]:
        #     mem_mask = mem_mask[:,:,:score.size()[-1]]
        norm_score = prob_normalize(score, mem_mask)
    output = attention_aggregate(value, norm_score)
    return output.squeeze(-2), norm_score.squeeze(-2)

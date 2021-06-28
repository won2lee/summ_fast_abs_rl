import math

import torch
from torch.nn import functional as F


#################### general sequence helper #########################
def len_mask(lens, device):
    """ users are resposible for shaping
    Return: tensor_type [B, T]
    """
    max_len = max(lens)
    batch_size = len(lens)
    mask = torch.ByteTensor(batch_size, max_len).to(device)
    mask.fill_(0)
    for i, l in enumerate(lens):
        mask[i, :l].fill_(1)
    return mask

def sequence_mean(sequence, seq_lens, dim=1):
    if seq_lens:
        assert sequence.size(0) == len(seq_lens)   # batch_size
        sum_ = torch.sum(sequence, dim=dim, keepdim=False)
        mean = torch.stack([s/l for s, l in zip(sum_, seq_lens)], dim=0)
    else:
        mean = torch.mean(sequence, dim=dim, keepdim=False)
    return mean

def sequence_loss(logits, targets, xent_fn=None, pad_idx=0):
    """ functional interface of SequenceLoss"""
    assert logits.size()[:-1] == targets.size()

    mask = targets != pad_idx
    target = targets.masked_select(mask)
    logit = logits.masked_select(
        mask.unsqueeze(2).expand_as(logits)
    ).contiguous().view(-1, logits.size(-1))
    if xent_fn:
        loss = xent_fn(logit, target)
    else:
        loss = F.cross_entropy(logit, target)
    assert (not math.isnan(loss.mean().item())
            and not math.isinf(loss.mean().item()))
    return loss


#################### LSTM helper #########################

def reorder_sequence(sequence_emb, order, batch_first=False):
    """
    sequence_emb: [T, B, D] if not batch_first
    order: list of sequence length
    """
    batch_dim = 0 if batch_first else 1
    assert len(order) == sequence_emb.size()[batch_dim]

    order = torch.LongTensor(order).to(sequence_emb.device)
    sorted_ = sequence_emb.index_select(index=order, dim=batch_dim)

    return sorted_

def reorder_lstm_states(lstm_states, order):
    """
    lstm_states: (H, C) of tensor [layer, batch, hidden]
    order: list of sequence length
    """
    assert isinstance(lstm_states, tuple)
    assert len(lstm_states) == 2
    assert lstm_states[0].size() == lstm_states[1].size()
    assert len(order) == lstm_states[0].size()[1]

    order = torch.LongTensor(order).to(lstm_states[0].device)
    sorted_states = (lstm_states[0].index_select(index=order, dim=1),
                     lstm_states[1].index_select(index=order, dim=1))

    return sorted_states

def get_sents_lenth(source, seq_lens, tgt = False):

    if type(source[0]) is not torch.Tensor and type(source[0]) is not list:
        source = [source]

    if type(source[0]) is list:
        print(f"source_len : {len(source)}")
        print(f"source_size() : {source[0].size()}")
    sbol = [4,5,6]

    # if seq_lens==[]:
    #     seq_lens = [len([k for k in s if k!=0]) for s in source]
    #src_len = [len(s) for s in source]     
    
    #   _<s>^  p  가  _ 계속 _ 오른 다 _  .  _ </s>
    #   1 0 2  0  0  1  0  1  0  0  1  0  1  0
    #for i,s in enumerate(source):
    #    print(f"k :{[k for k in s[:seq_lens[i]]]}")


    iXO = [[k.item() if k.item() in sbol else 0 for k in s[:seq_lens[i]]] for i,s in enumerate(source) ]
    #   1 0 2  0  0  1  0  1  0  0  1  0  1  0       <= XO
    #   0   2        5     7        10   12   [14]   <= XX1
    #     1       4     6        9    11    13       <= XX_R
    #   2,  3,       2,    3,       2     2          <= XX      sum(XX) == len(s)
    #   1 0 1  0  0  1  0  1  0  0  1 0   1  0
    #   1   2  0     1     1  0     1     1          <= XO (0~3 사이의 값)
    #   1,  2,       1,    2,       1     1          <= X_sub   sum(X_sub) == len(XO)
    XXi = [[i for i,v in enumerate(s) if v!=0]+[len(s)] for s in iXO]    # XX1
    #XX_R = [[k-1 for k in s[1:]] for s in XX]
    XX = [[s[i]-s[i-1] for i in range(1,len(s))] for s in XXi]     # index to interval lenth(어절의 길이)
    #XO = [ for i,k in enumerate(s) if k>0 or (k==0 and s[i+1] ==0]for s in XO]}
    XX_subtracted = [[k-1 if k>0 else 0 for k in s ] for s in XX]

    if tgt:
        XX_R = [[k-1 for k in s[1:]] for s in XXi]
        iXO = [[k for i,k in enumerate(s) if i not in XX_R[j]] for j,s in enumerate(iXO)]
        return XX, iXO, XX_subtracted  #, XK  # XX: Cutter, XO: lookup target list
 
    else:
        XX_len = [len(s) for s in XX] 
        return XX_len, XX, XX_subtracted

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from .util import reorder_sequence, reorder_lstm_states

#####################################################
if parallel => tokenize 단계에서 반영 

class parallel(nn.Module):
    def __init__(self, context_dim, state_dim, input_dim, bias=True):
        super().__init__()
        self.raw_emb_size = 
        self.embed_size = 
        self.sub_en_coder= nn.LSTM(self.raw_emb_size, self.embed_size)  #(embed_size, self.hidden_size)
        self.en_gate = nn.Linear(self.raw_emb_size, self.embed_size, bias=False) #(self.hidden_size, self.hidden_size, bias=False)
        self.sub_en_projection = nn.Linear(self.raw_emb_size, self.embed_size, bias=False)    #(self.hidden_size, self.hidden_size, bias=False) 

    def get_sents_lenth(source,seq_lens):
        sbol = [1,2,3,pad]
        if type(source[0]) is not list:
            source = [source]

        #    ^  p  가  _ 계속 _ 오른 다 _  .
        #    0         3     5       8    [10]       <=XX
        #    3         2     3       2               <=XX  XX_len = 4
        #    2         1     2       1               <=XX_subracted


        #source_lengths = [len(s) for s in source]
        #XX = [list(chain(*[[i,i+1] for i,k in enumerate(s) if k in sbol[0]])) for s in source]
        XXi= [[i for i,k in enumerate(s) if k in sbol] for s in source]
        XXi = [XXi[i] + [seq_lens[i]] for i in range(len(XXi))]   #len(XX): Batch size
        #to_sub = [[i for i,x in enumerate(xx) if x in to_add[j]] for j, xx in enumerate(XX)]
        XX = [[s[i]-s[i-1] for i in range(len(s)) if i>0] for s in XXi]     # index to interval lenth(어절의 길이)
        if len([s for s in XX if s==1]) > 0:
            print("There is 'empty word segment' !!!") 
        XX_len = [len(s) for s in XX]    # 문장의 어절 갯수
        XX_subtracted = [[x-1 if x>0 else 0 for x in xx ] for xx in XX]
        return XX_len, XX, XX_subtracted

    def parallel_encode(source,emb_sequence, seq_lens,embedding=None): #slang_is_tlang=False):

        if type(source[0]) is not list:
           source = [source]       

        source_lengths, Z, Z_sub = get_sents_lenth(source,seq_lens) # Z:각 sentence 내의 각 어절의 길이로 구성  list[list]
        #s_len = seq_lens #[len(s) for s in source]  # 원래의 문장 길이
        Z_len = [len(s) for s in Z]    # 문장의 어절 갯수


        def 
        max_Z = max(chain(*Z))  # 최대로 긴 어절
        
        
        max_l = max(seq_lens)           
        XX =  [s+[max_l-seq_lens[i]] if max_l>seq_lens[i] else s for i,s in enumerate(Z)] # total(interval lenth) to be source lenth 
        
        #src_padded = source # ? self.vocab.vocs.to_input_tensor(source, device=self.device)  

        X = list(chain(*[torch.split(sss,XX[i])[:Z_len[i]] for i,sss in enumerate(
            torch.split(source,1,-1))]))     #각 문장(batch)으로 자른 뒤 문장내 어절 단위로 자른다 

        X = pad_sequence(X).squeeze(-1)

        #if lang =='en':
        #    cap_id, len_X = get_X_cap(source, self.sbol)

        #X_embed = (embedding(sequence) if embedding is not None else sequence)    
        #X_embed = self.model_embeddings.vocabs(X)
        X_embed = embedding(X)

        out,(last_h1,last_c1) = self.sub_en_coder(X_embed)
        #X_proj = self.sub_en_projection(out[1:])               #sbol 부분 제거
        X_proj = self.sub_en_projection(X_embed[1:])
        X_gate = torch.sigmoid(self.en_gate(X_embed[1:]))


        X_way = self.dropout(X_gate * X_proj + (1-X_gate) * out[1:]) #X_proj)       

        #문장단위로 자르고 어절 단위로 자른 뒤 각 어절의 길이만 남기고 나머지는 버린 후 연결 (cat) 하여 문장으로 재구성         
        X_input = [torch.cat([ss[:Z_sub[i][j]]for j,ss in enumerate(
          torch.split(sss,1,1))],0) for i,sss in enumerate(torch.split(X_way,Z_len,1))]
        
        # 재구성된 문장의 길이가 다르기 때문에 패딩

        if encode:
            emb_sequence = pad_sequence(X_input).squeeze(-2)
            seq_lens = [sum([wl for wl in s]) for s in Z_sub]
            
            return emb_sequence, seq_lens
            
        else:
            emb_sequence = pad_sequence(X_input).squeeze(-2)[:-1]
            XO = [torch.tensor(x) for x in Z]
            XO = torch.tensor(pad_sequence(XO)).to(self.device) #,device = self.device) #<=[:-1]
            
            return emb_sequence, XO



def parallel_f(sequence, emb_seq, seq_l):
decoding 단계에서도 반영


#####################################################
def lstm_encoder(sequence, lstm,
                 seq_lens=None, init_states=None, embedding=None):
    """ functional LSTM encoder (sequence is [b, t]/[b, t, d],
    lstm should be rolled lstm)"""
    batch_size = sequence.size(0)
    if not lstm.batch_first:
        sequence = sequence.transpose(0, 1)
    emb_sequence = (embedding(sequence) if embedding is not None else sequence)
    # indent 하지 않은 것이 옳을 듯
    

    ##########################################################
    # parallel(emb_sequence, sequence)
    if parallel:
        emb_sequence, seq_lens = parallel_f(sequence,emb_sequence, seq_lens)
    ##########################################################


    if seq_lens:
        assert batch_size == len(seq_lens)
        sort_ind = sorted(range(len(seq_lens)),
                          key=lambda i: seq_lens[i], reverse=True)
        seq_lens = [seq_lens[i] for i in sort_ind]
        emb_sequence = reorder_sequence(emb_sequence, sort_ind,
                                        lstm.batch_first)

    if init_states is None:
        device = sequence.device
        init_states = init_lstm_states(lstm, batch_size, device)
    else:
        init_states = (init_states[0].contiguous(),
                       init_states[1].contiguous())

    if seq_lens:
        packed_seq = nn.utils.rnn.pack_padded_sequence(emb_sequence,
                                                       seq_lens)
        packed_out, final_states = lstm(packed_seq, init_states)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out)

        back_map = {ind: i for i, ind in enumerate(sort_ind)}
        reorder_ind = [back_map[i] for i in range(len(seq_lens))]
        lstm_out = reorder_sequence(lstm_out, reorder_ind, lstm.batch_first)
        final_states = reorder_lstm_states(final_states, reorder_ind)
    else:
        lstm_out, final_states = lstm(emb_sequence, init_states)

    return lstm_out, final_states


def init_lstm_states(lstm, batch_size, device):
    n_layer = lstm.num_layers*(2 if lstm.bidirectional else 1)
    n_hidden = lstm.hidden_size

    states = (torch.zeros(n_layer, batch_size, n_hidden).to(device),
              torch.zeros(n_layer, batch_size, n_hidden).to(device))
    return states


class StackedLSTMCells(nn.Module):
    """ stack multiple LSTM Cells"""
    def __init__(self, cells, dropout=0.0):
        super().__init__()
        self._cells = nn.ModuleList(cells)
        self._dropout = dropout

    def forward(self, input_, state):
        """
        Arguments:
            input_: FloatTensor (batch, input_size)
            states: tuple of the H, C LSTM states
                FloatTensor (num_layers, batch, hidden_size)
        Returns:
            LSTM states
            new_h: (num_layers, batch, hidden_size)
            new_c: (num_layers, batch, hidden_size)
        """
        hs = []
        cs = []
        for i, cell in enumerate(self._cells):
            s = (state[0][i, :, :], state[1][i, :, :])
            h, c = cell(input_, s)
            hs.append(h)
            cs.append(c)
            input_ = F.dropout(h, p=self._dropout, training=self.training)

        new_h = torch.stack(hs, dim=0)
        new_c = torch.stack(cs, dim=0)

        return new_h, new_c

    @property
    def hidden_size(self):
        return self._cells[0].hidden_size

    @property
    def input_size(self):
        return self._cells[0].input_size

    @property
    def num_layers(self):
        return len(self._cells)

    @property
    def bidirectional(self):
        return self._cells[0].bidirectional


class MultiLayerLSTMCells(StackedLSTMCells):
    """
    This class is a one-step version of the cudnn LSTM
    , or multi-layer version of LSTMCell
    """
    def __init__(self, input_size, hidden_size, num_layers,
                 bias=True, dropout=0.0):
        """ same as nn.LSTM but without (bidirectional)"""
        cells = []
        cells.append(nn.LSTMCell(input_size, hidden_size, bias))
        for _ in range(num_layers-1):
            cells.append(nn.LSTMCell(hidden_size, hidden_size, bias))
        super().__init__(cells, dropout)

    @property
    def bidirectional(self):
        return False

    def reset_parameters(self):
        for cell in self._cells:
            # xavier initilization
            gate_size = self.hidden_size / 4
            for weight in [cell.weight_ih, cell.weight_hh]:
                for w in torch.chunk(weight, 4, dim=0):
                    init.xavier_normal_(w)
            #forget bias = 1
            for bias in [cell.bias_ih, cell.bias_hh]:
                torch.chunk(bias, 4, dim=0)[1].data.fill_(1)

    @staticmethod
    def convert(lstm):
        """ convert from a cudnn LSTM"""
        lstm_cell = MultiLayerLSTMCells(
            lstm.input_size, lstm.hidden_size,
            lstm.num_layers, dropout=lstm.dropout)
        for i, cell in enumerate(lstm_cell._cells):
            cell.weight_ih.data.copy_(getattr(lstm, 'weight_ih_l{}'.format(i)))
            cell.weight_hh.data.copy_(getattr(lstm, 'weight_hh_l{}'.format(i)))
            cell.bias_ih.data.copy_(getattr(lstm, 'bias_ih_l{}'.format(i)))
            cell.bias_hh.data.copy_(getattr(lstm, 'bias_hh_l{}'.format(i)))
        return lstm_cell

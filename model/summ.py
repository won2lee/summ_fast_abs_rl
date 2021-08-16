import torch
from torch import nn
from torch.nn import init
from itertools import chain
from toolz.sandbox import unzip

from .rnn import lstm_encoder
from .rnn import MultiLayerLSTMCells
from .attention import step_attention
from .util import sequence_mean, len_mask, get_sents_lenth
from torch.nn.utils.rnn import pad_sequence #,pad_packed_sequence, pack_padded_sequence


INIT = 1e-2


class Seq2SeqSumm(nn.Module):
    def __init__(self, vocab_size, emb_dim,
                 n_hidden, bidirectional, n_layer, parallel, dropout=0.0, use_coverage=False):
        super().__init__()
        # embedding weight parameter is shared between encoder, decoder,
        # and used as final projection layer to vocab logit
        # can initialize with pretrained word vectors
        self.parallel = parallel
        self.use_coverage = use_coverage
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self._enc_lstm = nn.LSTM(
            n_hidden if self.parallel else emb_dim, n_hidden, n_layer,
            bidirectional=bidirectional, dropout=dropout
        )
        # initial encoder LSTM states are learned parameters
        state_layer = n_layer * (2 if bidirectional else 1)
        self._init_enc_h = nn.Parameter(
            torch.Tensor(state_layer, n_hidden)
        )
        self._init_enc_c = nn.Parameter(
            torch.Tensor(state_layer, n_hidden)
        )
        init.uniform_(self._init_enc_h, -INIT, INIT)
        init.uniform_(self._init_enc_c, -INIT, INIT)

        # vanillat lstm / LNlstm
        self._dec_lstm = MultiLayerLSTMCells(
            2 * n_hidden if parallel else 2*emb_dim, n_hidden, n_layer, dropout=dropout
            #n_hidden+emb_dim if parallel else 2*emb_dim, n_hidden, n_layer, dropout=dropout
        )
        # project encoder final states to decoder initial states
        enc_out_dim = n_hidden * (2 if bidirectional else 1)
        self._dec_h = nn.Linear(enc_out_dim, n_hidden, bias=False)
        self._dec_c = nn.Linear(enc_out_dim, n_hidden, bias=False)
        # multiplicative attention
        self._attn_wm = nn.Parameter(torch.Tensor(enc_out_dim, n_hidden))
        self._attn_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        init.xavier_normal_(self._attn_wm)
        init.xavier_normal_(self._attn_wq)
        # project decoder output to emb_dim, then
        # apply weight matrix from embedding layer
        self._projection = nn.Sequential(
            nn.Linear(2*n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, emb_dim, bias=False)  # emb_bin => n_hidden 으로 올리는 것 고려  
        )
        # functional object for easier usage

        # self._coverage = nn.Linear(2, 1, bias=False)

        if self.use_coverage:
            self.vT = nn.Linear(n_hidden, 1, bias=False)  
            self.enc_proj = nn.Linear(n_hidden, n_hidden, bias=False) 
            self.dec_proj = nn.Linear(n_hidden, n_hidden) # add bias for use_coverage
            self.w_cov = nn.Linear(1, n_hidden, bias=False) 
            self._coverage = (self.vT, self.enc_proj, self.dec_proj, self.w_cov) 

        self._decoder = AttentionalLSTMDecoder(
            self._embedding, self._dec_lstm,
            self._attn_wq, self._projection, cover=self._coverage if self.use_coverage else None
        )
        
        #self.parallel = parallel
        if self.parallel:
            self.sub_coder= nn.LSTM(emb_dim, n_hidden)  #(embed_size, self.hidden_size)
            self.sub_gate = nn.Linear(2*n_hidden, 1, bias=False) #(self.hidden_size, self.hidden_size, bias=False)
            #self.sub_gate = nn.Linear(emb_dim, n_hidden, bias=False) #(self.hidden_size, self.hidden_size, bias=False)
            self.sub_projection = nn.Linear(emb_dim, n_hidden, bias=False) 
            self.sub_dropout = nn.Dropout(p=0.2)

            self.target_ox_projection = nn.Linear(emb_dim+n_hidden, 4, bias=False) #emb_dim, 4, bias=False)
            self.copy_projection = nn.Linear(2*n_hidden, emb_dim, bias=False)
        


    def forward(self, article, art_lens, abstract):
        attention, init_dec_states, art_lens = self.encode(article, art_lens)
        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        logit = self._decoder((attention, mask), init_dec_states, abstract)
        return logit

    def encode(self, article, art_lens=None):
        size = (
            self._init_enc_h.size(0),
            len(art_lens) if art_lens else 1,
            self._init_enc_h.size(1)
        )
        init_enc_states = (
            self._init_enc_h.unsqueeze(1).expand(*size),
            self._init_enc_c.unsqueeze(1).expand(*size)
        )
        enc_art, final_states, art_lens = lstm_encoder(
            article, self._enc_lstm, art_lens,
            init_enc_states, self._embedding,
            parallel=self.parallel, 
            sub_module = ((self.sub_coder, self.sub_gate, self.sub_projection, self.sub_dropout) 
                                if self.parallel else None),
            paral_enc = Seq2SeqSumm.parallel_encode if self.parallel else None

        )   ################################################################# art_lens 추가 
        if self._enc_lstm.bidirectional:
            h, c = final_states
            final_states = (
                torch.cat(h.chunk(2, dim=0), dim=2),
                torch.cat(c.chunk(2, dim=0), dim=2)
            )
        init_h = torch.stack([self._dec_h(s)
                              for s in final_states[0]], dim=0)
        init_c = torch.stack([self._dec_c(s)
                              for s in final_states[1]], dim=0)
        init_dec_states = (init_h, init_c)
        attention = torch.matmul(enc_art, self._attn_wm).transpose(0, 1)
        init_attn_out = self._projection(torch.cat(
            [init_h[-1], sequence_mean(attention, art_lens, dim=1)], dim=1
        ))
        return attention, (init_dec_states, init_attn_out), art_lens  # art_lens 추가 

    def batch_decode(self, article, art_lens, go, eos, max_len):
        """ greedy decode support batching"""
        batch_size = len(art_lens)
        attention, init_dec_states = self.encode(article, art_lens)
        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        attention = (attention, mask)
        tok = torch.LongTensor([go]*batch_size).to(article.device)
        outputs = []
        attns = []
        states = init_dec_states
        for i in range(max_len):
            tok, states, attn_score = self._decoder.decode_step(
                tok, states, attention)
            outputs.append(tok[:, 0])
            attns.append(attn_score)
        return outputs, attns

    def decode(self, article, go, eos, max_len):
        attention, init_dec_states = self.encode(article)
        attention = (attention, None)
        tok = torch.LongTensor([go]).to(article.device)
        outputs = []
        attns = []
        states = init_dec_states
        for i in range(max_len):
            tok, states, attn_score = self._decoder.decode_step(
                tok, states, attention)
            if tok[0, 0].item() == eos:
                break
            outputs.append(tok[0, 0].item())
            attns.append(attn_score.squeeze(0))
        return outputs, attns

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)

    @staticmethod
    def parallel_encode(source,seq_lens,embedding,sub_module,tgt=False): #slang_is_tlang=False):

        if type(source[0]) is not torch.Tensor and type(source[0]) is not list:
           source = [source]       

        if tgt:
            Z, XO, Z_sub = get_sents_lenth(source,seq_lens,tgt)
        else:
            source_lengths, Z, Z_sub = get_sents_lenth(source,seq_lens) # Z:각 sentence 내의 각 어절의 길이로 구성  list[list]
        #s_len = seq_lens #[len(s) for s in source]  # 원래의 문장 길이
        Z_len = [len(s) for s in Z]    # 문장의 어절 갯수

        max_Z = max(chain(*Z))  # 최대로 긴 어절
        
        
        max_l = max(seq_lens)           
        XX =  [s+[max_l-seq_lens[i]] if max_l>seq_lens[i] else s for i,s in enumerate(Z)] # total(interval lenth) to be source lenth 
        
        #src_padded = source # ? self.vocab.vocs.to_input_tensor(source, device=self.device)  

        # print(f"source.size : {source.size()}")

        """
        for i,sss in enumerate(torch.split(source,1,0)):
            print(f"len of sss : {len(sss)}, size of sss: {sss.size()}")
            print(sum(XX[i]), len(XX[i]), Z_len[i])
            print(torch.split(sss,XX[i], -1)[:Z_len[i]])
        """

        X = list(chain(*[torch.split(sss.squeeze(0),XX[i])[:Z_len[i]] for i,sss in enumerate(
            torch.split(source,1,0))]))     #각 문장(batch)으로 자른 뒤 문장내 어절 단위로 자른다 
        # print(f"X[:4]: {[x.size() for x in X[:4]]}")    
        # print(f"pad_seqed_size: {pad_sequence(X).size()}")
        #X = pad_sequence(X)  #.squeeze(-1)
        X = pad_sequence(X).to(source.device) 
        

        #if lang =='en':
        #    cap_id, len_X = get_X_cap(source, self.sbol)

        #X_embed = (embedding(sequence) if embedding is not None else sequence)    
        #X_embed = self.model_embeddings.vocabs(X)
        X_embed = embedding(X)

        sub_coder, sub_gate,sub_projection,sub_dropout = sub_module

        out,(last_h1,last_c1) = sub_coder(X_embed)
        #X_proj = self.sub_en_projection(out[1:])               #sbol 부분 제거
        X_proj = sub_projection(X_embed[1:])
        #X_gate = torch.sigmoid(sub_gate(X_embed[1:]))
        X_gate = torch.sigmoid(sub_gate(torch.cat((X_proj,out[1:]),-1))) 


        X_way = sub_dropout(X_gate * X_proj + (1-X_gate) * out[1:]) #X_proj)       

        #문장단위로 자르고 어절 단위로 자른 뒤 각 어절의 길이만 남기고 나머지는 버린 후 연결 (cat) 하여 문장으로 재구성    
        for i,sss in enumerate(torch.split(X_way,Z_len,1)):
            if i> len(Z_sub)-1:
                print("###############list index out of range##################")
                print(i, len(Z_sub), Z_sub)
            for j,ss in enumerate(torch.split(sss,1,1)):
                if j> len(Z_sub[i])-1:
                    print("###############list index out of range##################")
                    print(i,j, len(Z_sub[i]), Z_sub)

        X_input = [torch.cat([ss[:Z_sub[i][j]] for j,ss in enumerate(
                   torch.split(sss,1,1))],0) for i,sss in enumerate(torch.split(X_way,Z_len,1))]
        # print(f"X-input[0].size() : {X_input[0].size()}")
        
        # 재구성된 문장의 길이가 다르기 때문에 패딩
            
        if tgt:
            emb_sequence = pad_sequence(X_input).squeeze(-2) #[:-1]
            XO = [torch.tensor(x) for x in XO]
            XO = torch.tensor(pad_sequence(XO)) #.to(self.device) #,device = self.device) #<=[:-1]
            
            return emb_sequence, XO.transpose(0,1)

        else:
            emb_sequence = pad_sequence(X_input).squeeze(-2)
            seq_lens = [sum([wl for wl in s]) for s in Z_sub]
            
            return emb_sequence, seq_lens

    def parallel_beam_code(self, X, init_vecs= None, device = 'cuda'): #slang_is_tlang=False):

        if type(X) is not torch.Tensor:
            X = torch.LongTensor(X)


        X = X.to(device)
        X_embed = self._embedding(X) #.contiguous()  #.unsqueeze(0)
        if len(X_embed.size()) <3:
            X_embed = X_embed.unsqueeze(0)
        # print(f"X_embed on cuda : {X_embed.is_cuda}, X_embed :{X_embed.size()}")
        if init_vecs:
            init_vecs = (init_vecs[0].contiguous(), init_vecs[1].contiguous())
            out,(h,c) = self.sub_coder(X_embed, init_vecs)
        else:
            out,(h,c) = self.sub_coder(X_embed)
        #X_proj = self.sub_en_projection(out[1:])               #sbol 부분 제거
        X_proj = self.sub_projection(X_embed)
        #X_gate = torch.sigmoid(self.sub_gate(X_embed))
        X_gate = torch.sigmoid(self.sub_gate(torch.cat((X_proj,out),-1))) 

        X_way = (X_gate * X_proj + (1-X_gate) * out).squeeze(0) #X_proj)  

        return X_way, (h,c)


class AttentionalLSTMDecoder(object):
    def __init__(self, embedding, lstm, attn_w, projection, parallel=False, sub_module=None, target_ox=None, copy_proj=None, cover=None):
        super().__init__()
        self._embedding = embedding
        self._lstm = lstm
        self._attn_w = attn_w
        self._projection = projection
        
        self.parallel = parallel
        self.vocab_size = self._embedding.weight.t().size()[-1]
        self.sub_module = sub_module
        # if parallel:
        #     self.sub_coder = sub_module[0], 
        #     self.sub_gate = sub_module[1], 
        #     self.sub_projection = sub_module[2], 
        #     self.sub_dropout = sub_module[3]
        self.target_ox_projection = target_ox 
        self.copy_projection = copy_proj
        self.coverage = cover


    def __call__(self, attention, init_states, target, tgt_lens):

        if self.parallel:
            target, XO = Seq2SeqSumm.parallel_encode(target,tgt_lens,self._embedding, self.sub_module, tgt=True)
            # target, XO = Seq2SeqSumm.parallel_encode(target,tgt_lens,self._embedding,
            #     sub_module = (self.sub_coder, self.sub_gate, self.sub_projection, self.sub_dropout), tgt=True)
            target = target.transpose(0,1)
            
        max_len = target.size(1)
        states = init_states
        logits = []
        coverage = [0.0]
        scores = []

        for i in range(max_len):
            tok = target[:, i:i+1]
            to_avoid = coverage[-1]
            logit, states, score = self._step(tok, states, attention, to_avoid)
            coverage.append(to_avoid + score if i>0 else score)
            scores.append(score)
            logits.append(logit) 


        # cover_lss = [torch.cat((cvr.unsqueeze(-2),scr.unsqueeze(-2)),-2).min(-2)[0] for cvr,scr in zip(coverage[1:-1],scores[1:])]
        # print(f"score : {scores[-1].size()},coverage : {coverage[-1].size()}")
        # print(f"cover_loss : {cover_lss[-1].size()},{cover_lss[-1].sum(-1).size()}")
        # cover_loss = [torch.cat((cvr,scr),-2).min(-2)[0].sum(-1) for cvr,scr in zip(coverage[1:-1],scores[1:])]
        cover_loss = [torch.cat((cvr.unsqueeze(-2),scr.unsqueeze(-2)),-2).min(-2)[0].sum(-1) for cvr,scr in zip(coverage[1:-1],scores[1:])]
        #print(f"cover_loss : {cover_loss[-1].size()}")

        if self.parallel:
            logits = list(unzip(logits))
            logit = [torch.stack(list(lgt), dim=1) for lgt in logits]
            return logit, XO, cover_loss

        logit = torch.stack(logits, dim=1)
        return logit, None, cover_loss

    # def __call__(self, attention, init_states, target):
    #     max_len = target.size(1)
    #     states = init_states
    #     logits = []
    #     for i in range(max_len):
    #         tok = target[:, i:i+1]
    #         logit, states, _ = self._step(tok, states, attention)
    #         logits.append(logit)
    #     logit = torch.stack(logits, dim=1)
    #     return logit

    def _step(self, tok, states, attention):
        prev_states, prev_out = states
        lstm_in = torch.cat(
            [self._embedding(tok).squeeze(1), prev_out],
            dim=1
        )
        states = self._lstm(lstm_in, prev_states)
        lstm_out = states[0][-1]
        query = torch.mm(lstm_out, self._attn_w)
        attention, attn_mask = attention
        # print(f"attention:{attention.size()},  attn_mask :{attn_mask.size()}, query : {query.size()}")
        context, score = step_attention(
            query, attention, attention, attn_mask)
        dec_out = self._projection(torch.cat([lstm_out, context], dim=1))
        states = (states, dec_out)
        logit = torch.mm(dec_out, self._embedding.weight.t())
        return logit, states, score

    def decode_step(self, tok, states, attention, to_avoid):
        logit, states, score = self._step(tok, states, attention, to_avoid)
        if self.parallel:
            out = [torch.max(logit[i], dim=1, keepdim=True)[1] for i in range(2)]
        else:
            out = torch.max(logit, dim=1, keepdim=True)[1]
        return out, states, score
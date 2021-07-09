import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .attention import step_attention
from .util import len_mask
from .summ import Seq2SeqSumm, AttentionalLSTMDecoder
from . import beam_search as bs


INIT = 1e-2


class _CopyLinear(nn.Module):
    def __init__(self, context_dim, state_dim, input_dim, bias=True):
        super().__init__()
        self._v_c = nn.Parameter(torch.Tensor(context_dim))
        self._v_s = nn.Parameter(torch.Tensor(state_dim))
        self._v_i = nn.Parameter(torch.Tensor(input_dim))
        init.uniform_(self._v_c, -INIT, INIT)
        init.uniform_(self._v_s, -INIT, INIT)
        init.uniform_(self._v_i, -INIT, INIT)
        if bias:
            self._b = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter(None, '_b')

    def forward(self, context, state, input_):
        output = (torch.matmul(context, self._v_c.unsqueeze(1))
                  + torch.matmul(state, self._v_s.unsqueeze(1))
                  + torch.matmul(input_, self._v_i.unsqueeze(1)))
        if self._b is not None:
            output = output + self._b.unsqueeze(0)
        return output

class CopySumm(Seq2SeqSumm):
    def __init__(self, vocab_size, emb_dim,
                 n_hidden, bidirectional, n_layer, parallel, dropout=0.0):
        super().__init__(vocab_size, emb_dim,
                         n_hidden, bidirectional, n_layer, parallel, dropout)
        self._copy = _CopyLinear(n_hidden, n_hidden, n_hidden+emb_dim if self.parallel else 2*emb_dim)
        # print(f"parallel : {parallel}")
        # print(f"self.parallel : {self.parallel}")

        if self.parallel:

            self._decoder = CopyLSTMDecoder(
                self._copy, self._embedding, self._dec_lstm,
                self._attn_wq, self._projection,
                parallel=self.parallel, 
                sub_module = (self.sub_coder, self.sub_gate, self.sub_projection, self.sub_dropout),
                target_ox=self.target_ox_projection, copy_proj=self.copy_projection
            )
        else:
            self._decoder = CopyLSTMDecoder(
                self._copy, self._embedding, self._dec_lstm,
                self._attn_wq, self._projection
            )  #emb_dim 추가하지 않음 

    def forward(self, article, art_lens, abstract, extend_art, extend_vsize,tgt_lens):
        attention, init_dec_states, art_lens = self.encode(article, art_lens) ####### add art_lens in return 
        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        logit,XO = self._decoder(
            (attention, mask, extend_art, extend_vsize),
            init_dec_states, abstract, tgt_lens 
        )
        return logit, XO

    def batch_decode(self, article, art_lens, extend_art, extend_vsize,
                     go, eos, unk, max_len):
        """ greedy decode support batching"""
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings
        attention, init_dec_states, _ = self.encode(article, art_lens)
        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        attention = (attention, mask, extend_art, extend_vsize)
        tok = torch.LongTensor([go]*batch_size).to(article.device)
        outputs = []
        attns = []
        states = init_dec_states

        
        _,sb_init = self.parallel_beam_code([[4,5,6]])
        print(f"sb_init : {sb_init.is_cuda}")
        init_vecs= ([sb_init[i][:,0].unsqueeze(1).expand((1,batch_size,sb_init.size()[-1])) 
            for i in range(2)])  # 초기 init_vector를 4('_')를 적용했을 떄를 값으로 

        for i in range(max_len):
            if self.parallel:
                tok, init_vecs = self.parallel_beam_code(self, tok, init_vecs) #slang_is_tlang=False):
                toks, states, attn_score = self._decoder.decode_step(
                    tok, states, attention)
                tok, xo = toks

                idx, init_h, init_c  = unzip([(i, sb_init[0][:,x].unsqueeze(1),sb_init[1][:,x].unsqueeze(1))
                                     for i,x in enumerate(xo) if x != 0])
                init_vecs[0][:,torch.tensor(idx)] = torch.cat(init_h,1)
                init_vecs[1][:,torch.tensor(idx)] = torch.cat(init_c,1)  #xo 값 에 따라 h,c update

                attns.append(attn_score)
                xos.append(xo)
                outputs.append(tok[:, 0].clone())
                tok.masked_fill_(tok >= vsize, unk)
            else:
                tok, states, attn_score = self._decoder.decode_step(
                    tok, states, attention)
                attns.append(attn_score)
                outputs.append(tok[:, 0].clone())
                tok.masked_fill_(tok >= vsize, unk)

        return (outputs, xos if self.parallel else None), attns

    def decode(self, article, extend_art, extend_vsize, go, eos, unk, max_len):
        vsize = self._embedding.num_embeddings
        attention, init_dec_states = self.encode(article)
        attention = (attention, None, extend_art, extend_vsize)
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
            if tok[0, 0].item() >= vsize:
                tok[0, 0] = unk
        return outputs, attns

    def batched_beamsearch(self, article, art_lens,
                           extend_art, extend_vsize,
                           go, eos, unk, max_len, beam_size, diverse=1.0):
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings
        attention, init_dec_states = self.encode(article, art_lens)
        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        all_attention = (attention, mask, extend_art, extend_vsize)
        attention = all_attention
        (h, c), prev = init_dec_states
        all_beams = [bs.init_beam(go, (h[:, i, :], c[:, i, :], prev[i]))
                     for i in range(batch_size)]
        finished_beams = [[] for _ in range(batch_size)]
        outputs = [None for _ in range(batch_size)]
        for t in range(max_len):
            toks = []
            all_states = []
            for beam in filter(bool, all_beams):
                token, states = bs.pack_beam(beam, article.device)
                toks.append(token)
                all_states.append(states)
            token = torch.stack(toks, dim=1)
            states = ((torch.stack([h for (h, _), _ in all_states], dim=2),
                       torch.stack([c for (_, c), _ in all_states], dim=2)),
                      torch.stack([prev for _, prev in all_states], dim=1))
            token.masked_fill_(token >= vsize, unk)

            topk, lp, states, attn_score = self._decoder.topk_step(
                token, states, attention, beam_size)

            batch_i = 0
            for i, (beam, finished) in enumerate(zip(all_beams,
                                                     finished_beams)):
                if not beam:
                    continue
                finished, new_beam = bs.next_search_beam(
                    beam, beam_size, finished, eos,
                    topk[:, batch_i, :], lp[:, batch_i, :],
                    (states[0][0][:, :, batch_i, :],
                     states[0][1][:, :, batch_i, :],
                     states[1][:, batch_i, :]),
                    attn_score[:, batch_i, :],
                    diverse
                )
                batch_i += 1
                if len(finished) >= beam_size:
                    all_beams[i] = []
                    outputs[i] = finished[:beam_size]
                    # exclude finished inputs
                    (attention, mask, extend_art, extend_vsize
                    ) = all_attention
                    masks = [mask[j] for j, o in enumerate(outputs)
                             if o is None]
                    ind = [j for j, o in enumerate(outputs) if o is None]
                    ind = torch.LongTensor(ind).to(attention.device)
                    attention, extend_art = map(
                        lambda v: v.index_select(dim=0, index=ind),
                        [attention, extend_art]
                    )
                    if masks:
                        mask = torch.stack(masks, dim=0)
                    else:
                        mask = None
                    attention = (
                        attention, mask, extend_art, extend_vsize)
                else:
                    all_beams[i] = new_beam
                    finished_beams[i] = finished
            if all(outputs):
                break
        else:
            for i, (o, f, b) in enumerate(zip(outputs,
                                              finished_beams, all_beams)):
                if o is None:
                    outputs[i] = (f+b)[:beam_size]
        return outputs


class CopyLSTMDecoder(AttentionalLSTMDecoder):
    def __init__(self, copy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._copy = copy


    def _step(self, tok, states, attention):
        prev_states, prev_out = states

        # lstm_in = torch.cat(
        #     [self._embedding(tok).squeeze(1), prev_out],
        #     dim=1
        # )

        #####################################################################
        if self.parallel==False:
            tok = self._embedding(tok)

        # lstm_in = torch.cat(
        #     [self._embedding(tok).squeeze(1), prev_out],
        #     dim=1
        # )
        # print(f"tok.size() : {tok.size()}, prev_out.size :{prev_out.size()}")
        lstm_in = torch.cat(
            [tok.squeeze(1), prev_out],
            dim=1
        )
        ######################################################################
        # print(f"lstm_in:{lstm_in.size()}, prev_states :{[tt.size() for tt in prev_states]}")
        states = self._lstm(lstm_in, prev_states)
        lstm_out = states[0][-1]
        query = torch.mm(lstm_out, self._attn_w)
        attention, attn_mask, extend_src, extend_vsize = attention
        context, score = step_attention(
            query, attention, attention, attn_mask
            )
        dec_out = self._projection(torch.cat([lstm_out, context], dim=1))

        # extend generation prob to extended vocabulary
        gen_prob = self._compute_gen_prob(dec_out, extend_vsize)
        # compute the probabilty of each copying
        copy_prob = torch.sigmoid(self._copy(context, states[0][-1], lstm_in))  #self._copy(context, states[0][-1], lstm_in))
        # add the copy prob to existing vocab distribution
        # print(f"context ; {(len(context),context[0].size()) if type(context) is list else context.size()}")
        # print(f"states[0][-1] ; {(len(states[0][-1]),states[0][-1][0].size()) if type(states[0][-1]) is list else states[0][-1].size()}")
        # print(f"lstm_in ; {(len(lstm_in),lstm_in[0].size()) if type(lstm_in) is list else lstm_in.size()}")
        # print(f"score ; {(len(score),score[0].size()) if type(score) is list else score.size()}")
        # print(f"dec_out ; {(len(dec_out),dec_out[0].size()) if type(dec_out) is list else dec_out.size()}")
        # print(f"gen_prob ; {(len(gen_prob),gen_prob[0].size()) if type(gen_prob) is list else gen_prob.size()}")
        # print(f"copy_prob ; {(len(copy_prob),copy_prob[0].size()) if type(copy_prob) is list else copy_prob.size()}")
        # print(f"extend_src ; {(len(extend_src),extend_src[0].size()) if type(extend_src) is list else extend_src.size()}")
        """
        context ; torch.Size([32, 256])
        score ; torch.Size([32, 46])
        dec_out ; torch.Size([32, 128])
        gen_prob ; torch.Size([32, 30550])
        copy_prob ; torch.Size([32, 1])
        extend_src ; torch.Size([32, 86])
        """
        lp = torch.log(
            ((-copy_prob + 1) * gen_prob
            ).scatter_add(
                dim=1,
                index=extend_src.expand_as(score),
                src=score * copy_prob
        ) + 1e-8)  # numerical stability for log

        if self.parallel:
            lp2 = F.log_softmax(self.target_ox_projection(torch.cat((
                dec_out, 
                self.copy_projection(torch.cat((context, states[0][-1]),-1))
                ),-1)),-1)
            # lp2 = self.target_ox_projection(
            #     (-copy_prob + 1) * dec_out 
            #     + copy_prob * self.copy_projection(torch.cat((context, states[0][-1]),-1))
            #     )
        else:
            lp2 = None
        return (lp, lp2), (states, dec_out), score


    def topk_step(self, tok, states, attention, k):
        """tok:[BB, B], states ([L, BB, B, D]*2, [BB, B, D])"""
        (h, c), prev_out = states

        # lstm is not bemable
        nl, _, _, d = h.size()
        beam, batch = tok.size()
        lstm_in_beamable = torch.cat(
            [self._embedding(tok), prev_out], dim=-1)
        lstm_in = lstm_in_beamable.contiguous().view(beam*batch, -1)
        prev_states = (h.contiguous().view(nl, -1, d),
                       c.contiguous().view(nl, -1, d))
        h, c = self._lstm(lstm_in, prev_states)
        states = (h.contiguous().view(nl, beam, batch, -1),
                  c.contiguous().view(nl, beam, batch, -1))
        lstm_out = states[0][-1]

        # attention is beamable
        query = torch.matmul(lstm_out, self._attn_w)
        attention, attn_mask, extend_src, extend_vsize = attention
        context, score = step_attention(
            query, attention, attention, attn_mask)
        dec_out = self._projection(torch.cat([lstm_out, context], dim=-1))

        # copy mechanism is not beamable
        gen_prob = self._compute_gen_prob(
            dec_out.contiguous().view(batch*beam, -1), extend_vsize)
        copy_prob = torch.sigmoid(
            self._copy(context, lstm_out, lstm_in_beamable)
        ).contiguous().view(-1, 1)
        lp = torch.log(
            ((-copy_prob + 1) * gen_prob
            ).scatter_add(
                dim=1,
                index=extend_src.expand_as(score).contiguous().view(
                    beam*batch, -1),
                source=score.contiguous().view(beam*batch, -1) * copy_prob
        ) + 1e-8).contiguous().view(beam, batch, -1)

        k_lp, k_tok = lp.topk(k=k, dim=-1)
        return k_tok, k_lp, (states, dec_out), score

    def _compute_gen_prob(self, dec_out, extend_vsize, eps=1e-6):
        logit = torch.mm(dec_out, self._embedding.weight.t())
        bsize, vsize = logit.size()
        if extend_vsize > vsize:
            ext_logit = torch.Tensor(bsize, extend_vsize-vsize
                                    ).to(logit.device)
            ext_logit.fill_(eps)
            gen_logit = torch.cat([logit, ext_logit], dim=1)
        else:
            gen_logit = logit
        gen_prob = F.softmax(gen_logit, dim=-1)
        return gen_prob

    def _compute_copy_activation(self, context, state, input_, score):
        copy = self._copy(context, state, input_) * score
        return copy

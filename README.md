
# Summ Model 
#### Modified Model based on      
### Fast_Abs_RL Model (원 모델) 
- [Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting (Chen and Bansal, 2018)](https://arxiv.org/abs/1805.11080)
- Source Code : [https://github.com/ChenRocks/fast_abs_rl.git](https://github.com/ChenRocks/fast_abs_rl.git)
---------------------------------------------------------------------
#### 3 Main Training Processes in Fast_Abs_RL Model (원 모델)
- Extractor   (train_extractor_ml.py) :     
... article의 문장 중 주요 문장을 추출 (문장 분석 convolution, 문장 간 분석: LSTM)
- Abstractor  (train_abstractor.py) :     
... 추출된 문장을 1 to 1으로 요약     
... based on seq-to-seq attention model + copy mechanism
- Reinforce-guided extraction  (train_full_rl.py) :    
... Extractor로 문장 추출 - Abstractor로 요약 (1 to 1) - rouge score - 문장 추출에 feedback

#### Added and Modified in this Modified Model (본 요약 모델에서 추가/수정된 부분) 
- added:    
... sub-module (한글은 어절 단위로, 영어는 단어 단위로 토큰을 연계해 주는 역할)   
... coverage-loss (중복된 단어 생성을 억제하기 위해 누적 어텐션 스코어가 큰 토큰을 다시 어텐션 할 경우 페널티)
- modified:    
... self-developed tokenizer   
... 1 to 1 요약 (1 extracted -> 1 abstract)을  n to 1, n to k 요약으로 (전체 문장을 한개 혹은 k개 문장으로) 확장  

-------------------------------------------------------------------------    
### fast_abs_rl Model (원 모델) vs. Modified Model 
#### 원 모델 개요도 
<img src="/images/fast_abs_rl.jpg" width="700px" title="모델 개요도" alt="fast_abs_rl"></img><br/>

#### 본 모델 개요도 
<img src="/images/modfied_fast_abs_rl.jpg" width="700px" title="본 모델 개요도" alt="modified_fast_abs_rl"></img><br/>

    ❶❹❺ : modified (blue-colored number in the above figure)   ②③ : added (red-colored number)

    ❶  tokenizer : self-developed mainly for korean language 
                   (https://github.com/won2lee/preProc.git)    
    
    ②  sub-module :      
          ... 한글 어절을 구분하여 어절 안에 있는 토큰 들의 의미와 기능을 연결 하는 LSTM sub-module,      
          ... 영어의 경우 단어 단위로 토큰 구분하고 연결하며 및 대소 문자를 구분하는 기능     

    ③  coverage-mechanism     
          ... 단어 중복 생성 방지를 위해 이미 누적 attention score 가 높은 토큰을 다시 선택하지 않도록 유도 

    ❹  abstraction option added : n (extracted) to 1 (big abstract) 
          ... 추출(extracted)된 모든 문장을 한개의 문장으로 요약한 AIHUB의 한글 데이터세트를 사용하기 위한 옵션

    ❺  reward option added
          ... reward in fast_abs_rl model
              opt 0 : n_th extraction -> n_th abstract (1 to 1 ROUGE score):
                        ROUGE(n_th abst, n_th target),
                        where n_th abst is 1 abstract sentence from n_th extracted sentence
          ... added options in this model
              opt 1 : n_th extraction -> increased ROUGE score : 
                        ROUGE([1:n] abst, 1 big target) - ROUGE([1:n-1] abst, 1 big target), 
                        where [1:n] abst is 1 abstract sentence from [1:n] extracted sentences,
                           i big target is concat([1:n] target sentences) 
                           or 1 target sentence covering all sentences of the current article
              opt 2 : n_th extraction -> ROUGE 의 증가분 = ROUGE([n_th abst, 1 (big target))
                         

------------------------------------------------------------------
### To Train
#### My Colab Example

*[here](https://github.com/ChenRocks/cnn-dailymail)*
for downloading and preprocessing the CNN/DailyMail dataset.

    !pip install cytoolz
    !pip install pyrouge
    !pip install tensorboardX

    import os
    os.environ["DATA"]="/content/fast_abs_rl/cnn-dailymail/finished_files"
    os.environ["ROUGE"] = "/content/fast_abs_rl/pyrouge/tools/ROUGE-1.5.5"

1. preprocess for this Modified Model
```
!python3 temp_preproc.py --in_path=../cnn-dailymail/finished_files/test-origin/ --out_path=../cnn-dailymail/finished_files/test/ --lang=en
!python3 make_vocab_file.py --in_path=../cnn-dailymail/finished_files/train/ --out_path=../cnn-dailymail/finished_files/ 
```
2. pretrain a *word2vec* word embedding
```
!python3 train_word2vec.py --path=pathto/word2vec/ --dim=128
```
3. make the pseudo-labels
```
!python3 make_extraction_labels.py
```
4. train *abstractor* and *extractor* using ML objectives
```
!python3 train_abstractor.py --path=pathto/abstractor/model --w2v=/content/fast_abs_rl/pathto/word2vec/word2vec.128d.51k.bin --emb_dim=128 --max_abs=45 --ckpt_freq=1500 --lr=0.00002 --n_layer=1  --decay=0.7 --lr_p=1 --parallel --lang=en --use_coverage (--continued)
!python3 train_extractor_ml.py --path=pathto/extractor/model  --w2v=pathto/word2vec/word2vec.128d.198k.bin --emb_dim=128 --ckpt_freq=1500 --lr=0.00007 --max_word=150 --lr_p=1 --lang=en --reverse_parallel (--continued)
```
5. train the *full RL model*
```
!python3 train_full_rl.py --path=pathto/save/model --abs_dir=pathto/abstractor/model --ext_dir=pathto/extractor/model --batch=16 --lr=0.0001 --lr_p=1 --decay=0.7 --patience=10 --ckpt_freq=500 --gamma=0.95 --reverse_parallel (--continued)
```

After the training finishes you will be able to run the decoding and evaluation following the instructions in the previous section.

    !python3 decode_full_model.py --path=pathto/save/decoded/files --model_dir=pathto/save/model --max_dec_word=45 --parallel --beam=5 --test --reverse_parallel
    !python3 make_eval_references.py
    !python3 eval_full_model.py --rouge --decode_dir=pathto/save/decoded/files
  

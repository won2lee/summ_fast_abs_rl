
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

    ❶❹❺ : modified (blue-colored number in the above figure)   ❷❸ : added (red-colored number)

    ❶  tokenizing and embedding    
          ... tokenizer : 직접 개발 (https://github.com/won2lee/preProc.git)     
          ... embedding : pretrained on 첨부 1 번역모델 (https://github.com/won2lee/KorEn_NMT.git)      
    ❷  sub-module :      
          ... 한글 어절을 구분하여 어절 안에 있는 토큰 들의 의미와 기능을 연결 하는 LSTM sub-module,      
          ... 영어의 경우 단어 단위로 토큰 구분하고 연결하며 및 대소 문자를 구분하는 기능     

    ❸  coverage-mechanism     
          ... 단어 중복 생성 방지를 위해 이미 누적 attention score 가 높은 토큰을 다시 선택하지 않도록 유도 

    ❹  n (extracted) to 1 (big abstract) : option   
          ... 추출(extracted) 된 모든 문장을 한개의 문장으로 요약한 AIHUB의 한글 요약 데이터세트를 사용하기 위한 옵션

    ❺  reward option added
          . origin model
          ... n_th extraction -> n_th abstract => ROUGE(n_th abst, n_th target)         
          . added options
          ... opt 1 : n_th extraction -> ROUGE 의 증가분 = ROUGE([1:n] abst, 1 (big target)) - ROUGE([1:n-1] abst, 1 (big target)) 
          ......where [1:n] abst is 1 abstracted sentence from [1:n] extracted sentences
          ... opt 2 : n_th extraction -> ROUGE 의 증가분 = ROUGE([n] abst, 1 (big target))
          ......where [n] abst is 1 abstracted sentence from n_th extracted sentence   

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
    
The above will use the best hyper-parameters we used in the paper as default.
Please refer to the respective source code for options to set the hyper-parameters.


# Fast Abstractive Summarization-RL
This repository contains the code for our ACL 2018 paper:

*[Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting](https://arxiv.org/abs/1805.11080)*.

You can
1. Look at the generated summaries and evaluate the ROUGE/METEOR scores
2. Run decoding of the pretrained model
3. Train your own model

If you use this code, please cite our paper:
```
@inproceedings{chen2018fast,
  title={Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting},
  author={Yen-Chun Chen and Mohit Bansal},
  booktitle={Proceedings of ACL},
  year={2018}
}
```

## Dependencies
- **Python 3** (tested on python 3.6)
- [PyTorch](https://github.com/pytorch/pytorch) 0.4.0
    - with GPU and CUDA enabled installation (though the code is runnable on CPU, it would be way too slow)
- [gensim](https://github.com/RaRe-Technologies/gensim)
- [cytoolz](https://github.com/pytoolz/cytoolz)
- [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)
- [pyrouge](https://github.com/bheinzerling/pyrouge) (for evaluation)

You can use the python package manager of your choice (*pip/conda*) to install the dependencies.
The code is tested on the *Linux* operating system.

## Evaluate the output summaries from our ACL paper
Download the output summaries *[here](https://bit.ly/acl18_results)*.

To evaluate, you will need to download and setup the official ROUGE and METEOR
packages.

We use [`pyrouge`](https://github.com/bheinzerling/pyrouge)
(`pip install pyrouge` to install)
to make the ROUGE XML files required by the official perl script.
You will also need the official ROUGE package.
(However, it seems that the original ROUGE website is down.
An alternative can be found
*[here](https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5)*.)
Please specify the path to your ROUGE package by setting the environment variable
`export ROUGE=[path/to/rouge/directory]`.


For METEOR, we only need the JAR file `meteor-1.5.jar`.
Please specify the file by setting the environment variable
`export METEOR=[path/to/meteor/jar]`.

Run
```
python eval_acl.py --[rouge/meteor] --decode_dir=[path/to/decoded/files]
```
to get the ROUGE/METEOR scores reported in the paper.

## Decode summaries from the pretrained model
Download the pretrained models *[here](https://bit.ly/acl18_pretrained)*.
You will also need a preprocessed version of the CNN/DailyMail dataset.
Please follow the instructions
*[here](https://github.com/ChenRocks/cnn-dailymail)*
for downloading and preprocessing the CNN/DailyMail dataset.
After that, specify the path of data files by setting the environment variable
`export DATA=[path/to/decompressed/data]`

We provide 2 versions of pretrained models.
Using `acl` you can reproduce the results reported in our paper.
Using `new` you will get our latest result trained with a newer version of PyTorch library
which leads to slightly higher scores.

To decode, run
```
python decode_full_model.py --path=[path/to/save/decoded/files] --model_dir=[path/to/pretrained] --beam=[beam_size] [--test/--val]
```
Options:
- beam_size: number of hypothesis for (diverse) beam search. (use beam_size > 1 to enable reranking)
  - beam_szie=1 to get greedy decoding results (rnn-ext + abs + RL)
  - beam_size=5 is used in the paper for the +rerank model (rnn-ext + abs + RL + rerank)
- test/val: decode on test/validation dataset

If you want to evaluate on the generated output files,
please follow the instructions in the above section to setup ROUGE/METEOR.

Next, make the reference files for evaluation:
```
python make_eval_references.py
```
and then run evaluation by:
```
python eval_full_model.py --[rouge/meteor] --decode_dir=[path/to/save/decoded/files]
```

### Results
You should get the following results

Validation set

| Models             | ROUGEs (R-1, R-2, R-L) | METEOR |
| ------------------ |:----------------------:| ------:|
| **acl** |
| rnn-ext + abs + RL | (41.01, 18.20, 38.57)  |  21.10 |
| + rerank           | (41.74, 18.39, 39.40)  |  20.45 |
| **new** |
| rnn-ext + abs + RL | (41.23, 18.45, 38.71)  |  21.14 |
| + rerank           | (42.06, 18.80, 39.68)  |  20.58 |

Test set

| Models             | ROUGEs (R-1, R-2, R-L) | METEOR |
| ------------------ |:----------------------:| ------:|
| **acl** |
| rnn-ext + abs + RL | (40.03, 17.61, 37.58)  |  21.00 |
| + rerank           | (40.88, 17.81, 38.53)  |  20.38 |
| **new** |
| rnn-ext + abs + RL | (40.41, 17.92, 37.87)  |  21.13 |
| + rerank           | (41.20, 18.18, 38.79)  |  20.56 |

**NOTE**:
The original models in the paper are trained with pytorch 0.2.0 on python 2. 
After the acceptance of the paper, we figured it is better for the community if
we release the code with latest libraries so that it becomes easier to build new
models/techniques on top of our work. 
This results in a negligible difference w.r.t. our paper results when running the old pretrained model;
and gives slightly better scores than our paper if running the new pretrained model.

## Train your own models
Please follow the instructions
*[here](https://github.com/ChenRocks/cnn-dailymail)*
for downloading and preprocessing the CNN/DailyMail dataset.
After that, specify the path of data files by setting the environment variable
`export DATA=[path/to/decompressed/data]`

To re-train our best model:
1. pretrained a *word2vec* word embedding
```
python train_word2vec.py --path=[path/to/word2vec]
```
2. make the pseudo-labels
```
python make_extraction_labels.py
```
3. train *abstractor* and *extractor* using ML objectives
```
python train_abstractor.py --path=[path/to/abstractor/model] --w2v=[path/to/word2vec/word2vec.128d.226k.bin]
python train_extractor_ml.py --path=[path/to/extractor/model] --w2v=[path/to/word2vec/word2vec.128d.226k.bin]
```
4. train the *full RL model*
```
python train_full_rl.py --path=[path/to/save/model] --abs_dir=[path/to/abstractor/model] --ext_dir=[path/to/extractor/model]
```
After the training finishes you will be able to run the decoding and evaluation following the instructions in the previous section.

The above will use the best hyper-parameters we used in the paper as default.
Please refer to the respective source code for options to set the hyper-parameters.


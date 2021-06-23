import argparse
import json
import re
import os
from os.path import join, exists
from tqdm.notebook import tqdm
from itertools import chain
from cytoolz import curry
from utils import to_start, rid_blank, preproc_num, to_normal
from trns.preproc_En import pre_en, preproc_en
from trns.preproc_kor import preproc_ko2en

@curry
def preProc(lang, to_start, pre_ko, preproc_en, en_vocs, X):
    
    #X = to_start(X)
    
    if lang == 'en':
        #X = pre_en(X)
        X = preproc_en(X,en_vocs)
        X = preproc_num(X)
       
    else:
        X = pre_ko.forward(X)
        X = preproc_num(X) 

    return X 


def preProc_save(X, step_num, lang,to_start, pre_ko, preproc_en, en_vocs,f_toSave):
    
    #n_iter = len(X) // step_num +1
    n_iter = len(X) // step_num + min((len(X) % step_num), 1) * 1

    for i in tqdm(range(n_iter)):
        Xsub = X[i*step_num:(i+1)*step_num]
        #Y = list(chain(*[preProc(s, 'en', to_start, pre_ko, preproc_en, en_vocs) for s in Xsub]))
        Y = preProc(Xsub, lang, to_start, pre_ko, preproc_en, en_vocs)
        with open(f_toSave, 'w' if i==0 else 'a') as f:
            f.write('\n'.join(Y)+'\n')
    return Y

#with open('to_preproc/pre_processed_short05.en', 'r') as f:
#    X = f.read().split('\n')[:100]


def sanitize_input(in_file, dataList = None):
    
    p = re.compile('[\`\^]')
    p1 = re.compile('\`')
    p2 = re.compile('\^')
    
    if in_file:
        with open(in_file, 'r') as f:
            X = f.read().split('\n')
            X = [s+'.' for s in X]
    elif dataList:
        X = dataList

    if len([s for s in X if p.search(s) is not None])>0:
        print("`^ id detected !!!!")
        print([(i,s) for i,s in enumerate(X) if p.search(s) is not None])

    X = [p2.sub('ˆ',p1.sub("'",s)) for s in X] #if p.search(s) is None]
    
    return X

def fast_preproc(in_path,out_path, lang):
    from glob import glob
    f_list = glob(in_path+"*")
    
    if not exists(out_path):
        os.makedirs(out_path)
     
    pre_ko = preproc_ko2en()
    en_vocs = pre_en()
    preproc = preProc(lang, to_start, pre_ko, preproc_en, en_vocs)
    
    for i,fi in enumerate(f_list):
        with open(fi) as f:
            js = json.loads(f.read())
        for k in ["article", "abstract"]:
            js[k] = preproc(sanitize_input(None, js[k]))
        with open(join(out_path, fi.split('/')[-1]),"w") as f:
            json.dump(js,f,indent=4) 
        if i%10000==0:
            print(f"{i}th file was done") 

def main_proc(args):
    
    if not exists(args.path):
        os.makedirs(args.path)
        
    pre_ko = preproc_ko2en()
    en_vocs = pre_en()

    path0 = args.path #'sample_data/'
    f_toSave = path0+args.output+'.'
    step_num = 10000
    
    lang = args.lang
    X = sanitize_input(path0+args.input)
    Y = preProc_save(step_num, lang, to_start, pre_ko, preproc_en, en_vocs,f_toSave+lang,X)

    #X = sanitize_input(path0+'inputs/inX.ko')
    #Y = preProc_save(X, step_num, lang, to_start, pre_ko, preproc_en, en_vocs,f_toSave+'ko')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='program to preproceed')
    parser.add_argument('--path', default='sample_data/', help='root of the data')

    # model options
    parser.add_argument('--lang', action='store', default='ko',
                        help='language of input data: en or ko')
    parser.add_argument('--input', action='store', default='inputs/inX.ko',
                        help='input data for preprocess')
    parser.add_argument('--output', action='store', default='outputs/outX',
                        help='out data after proprocess')

    args = parser.parse_args()
    
    main_proc(args)

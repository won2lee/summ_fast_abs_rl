""" make reference text files needed for ROUGE evaluation """
import json
import os
from os.path import join, exists
from time import time
from datetime import timedelta

from utils import count_data
from decoding import make_html_safe
from decode_full_model import for_cnn

try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')


def dump(split):
    start = time()
    print('start processing {} split...'.format(split))
    data_dir = join(DATA_DIR, split)
    dump_dir = join(DATA_DIR, 'refs', split)

    from glob import glob
    f_list = glob(data_dir+"/*")
    n_data = len(f_list)
    # n_data = count_data(data_dir)
    #for i in range(n_data):
    for i,fn in enumerate(f_list):
        print('processing {}/{} ({:.2f}%%)\r'.format(i, n_data, 100*i/n_data),
              end='')
        with open(join(data_dir, '{}'.format(fn.split('/')[-1]))) as f:
            data = json.loads(f.read())
        abs_sents = [for_cnn(''.join(s.split()).strip()) for s in data['abstract']]
        with open(join(dump_dir, '{}.ref'.format(i)), 'w') as f:
            f.write(make_html_safe('\n'.join(abs_sents)))
    print('finished in {}'.format(timedelta(seconds=time()-start)))

def main():
    for split in ['val', 'test']:  # evaluation of train data takes too long
        if not exists(join(DATA_DIR, 'refs', split)):
            os.makedirs(join(DATA_DIR, 'refs', split))
        dump(split)

if __name__ == '__main__':
    main()

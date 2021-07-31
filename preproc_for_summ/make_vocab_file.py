import argparse
import json
import collections
import os
import pickle as pkl
from glob import glob

def main(args):
    in_path = args.in_path
    f_list = glob(in_path+"*")
    to_load = True

    print(len(f_list))

    if len(f_list) == 1:
        to_load = False
        with open(f_list[0]) as f:
            f_list = json.loads(f.read())

    finished_files_dir =args.out_path
    vocab_counter = collections.Counter()

    for i,fi in enumerate(f_list):
        #if i>100:
        #    break
        if to_load:
            with open(fi) as f:
                js = json.loads(f.read())
        else:
            js = fi

        art_tokens = ' '.join(js['article']).split()
        abs_tokens = ' '.join(js['abstract']).split()
        tokens = art_tokens + abs_tokens
        tokens = [t.strip() for t in tokens] # strip
        tokens = [t for t in tokens if t != ""] # remove empty
        vocab_counter.update(tokens)

    print("Writing vocab file...")
    with open(os.path.join(finished_files_dir, "vocab_cnt.pkl"),'wb') as vocab_file:
        pkl.dump(vocab_counter, vocab_file)
    print("Finished writing vocab file")  

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='program to preproceed')
    parser.add_argument('--in_path', default='in_path/', help='root of the data')
    parser.add_argument('--out_path', default="finished_files_dir/",
                        help='out data after proprocess')

    args = parser.parse_args()
    
    main(args)

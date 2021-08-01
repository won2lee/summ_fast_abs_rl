import argparse
#import json
#import collections
#import os
import shutil
#import pickle as pkl
from glob import glob

def main(args):
    pf_list = glob(args.in_path+"*")
    f_list = [f.split('/')[-1] for f in pf_list]
    ibest = sorted([(i,f) for i,f in enumerate(f_list)], key=lambda x:x[1])[0][0]
    print(pf_list[ibest])

    shutil.copy(pf_list[ibest],args.out_path)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='program to preproceed')
    parser.add_argument('--in_path', default='/content/fast_abs_rl/pathto/abstractor/model/ckpt/', help='root of the data')
    parser.add_argument('--out_path', default="/content/drive/MyDrive/default/",
                        help='destination')

    args = parser.parse_args()
    
    main(args)

# python3 save_in_drive.py --out_path="/content/drive/MyDrive/fast_abs_folder/abst/ckpt/ckpt_no_grad"
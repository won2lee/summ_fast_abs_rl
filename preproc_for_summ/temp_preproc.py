import argparse
from data_preproc import fast_preproc

def main(args):
    fast_preproc(args.in_path, args.out_path, args.lang)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='data for kor_summ'
    )
    parser.add_argument('--in_path', required=True, help='root of input data')
    parser.add_argument('--out_path', required=True, help='path of outputs')
    parser.add_argument('--lang', type=str, action='store', default='en',
                        help='language articles written in')
    args = parser.parse_args()
    main(args)
    
    ## python temp_preproc.py --in_path=../test_path/in/ --out_path=../test_path/out/ --lang=ko
    # python temp_preproc.py --in_path=/content/fast_abs_rl/corea_dailynews/finished_files/news_valid_mini.json \
    # --out_path=/content/fast_abs_rl/corea_dailynews/finished_files/val --lang=ko
from data_preproc import fast_preproc

def main():
    in_path = "/content/fast_abs_rl/cnn-dailymail/finished_files/train/"
    out_path = "/content/fast_abs_rl/cnn-dailymail/finished_files/train_new/"
    lang = "en"

    fast_preproc(in_path,out_path, lang)

if __name__ == "__main__":
    main()
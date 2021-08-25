from glob import glob
import json
import torch
import numpy as np

def make_new_fileset():
    in_path = "finished_files/train/"
    out_path = "mono_abs_train_small2/"
    flist = glob(in_path +"*")
    new_flist = []
    ext_snts = []
    abs_snts = []
    for fn in flist[:100]:
        jd = json.load(open(fn,"r"))
        art = jd['article']
        ext = [art[ix].split() for ix in jd['extracted']]
        abss = [s.split() for s in jd["abstract"]][0]
        i_match = sorted([(i,len(set(s) & set(abss))) for i,s in enumerate(ext)], key=lambda x:x[1], reverse=True)[0]
        if len(ext[i_match[0]]) > 1.2*len(abss) and len(set(ext[i_match[0]]) & set(abss)) / len(set(abss)) >0.6:
            new_flist.append(fn)
            ext_snts.append(ext[i_match[0]])
            abs_snts.append(abss)
            jd["extracted"] = [jd["extracted"][i_match[0]]]
            json.dump(jd, open(out_path+fn.split('/')[-1],"w"), ensure_ascii=False,indent=4)

    ext_snt_len = [len(s) for s in ext_snts]
    abs_snt_len = [len(s) for s in abs_snts]   
    print(f"extracted sent_len  : mean = {np.mean(ext_snt_len)},  std = {np.std(ext_snt_len)}")
    print(f"abstracted sent_len : mean = {np.mean(abs_snt_len)},  std = {np.std(abs_snt_len)}")
    print(f"tot num of flist      : {len(flist)}")
    print(f"num of selected flist : {len(new_flist)}")
    
if __name__ == '__main__':
    make_new_fileset()

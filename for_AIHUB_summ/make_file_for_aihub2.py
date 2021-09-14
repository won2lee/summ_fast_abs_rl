from glob import glob
import json
import torch
import re
import numpy as np
from collections import Counter
from cytoolz import curry, concat

@curry
def to_sep(s,p):
    return p[0].sub('_',p[1].sub('',s)).strip().split('_')
sepf= to_sep(p=[re.compile("[\^\`]"), re.compile("\s+")])
cwords = set("' . ' ( ) ·".split()+[""])
print(f"cwords :{cwords}")

def make_new_fileset(to_shuffle=False, to_cut=False):
    in_path = "/content/fast_abs_rl/corea_dailynews/finished_files/"
    out_path = "/content/fast_abs_rl/corea_dailynews/finished_files/"

    k_uniq = 1


    """
    block to extract common words
    counter = Counter()
    for fn in flist[:1000]:
        jd = json.load(open(fn,"r"))["abstract"][0]
        counter+=Counter(sepf(jd))
    json.dump(counter,open("counter_small.json","w"),ensure_ascii=False,indent=4)
    print(counter.most_common(100))
    """
    for dset in ["val", "train"]:
        if to_shuffle and dset=='val':
            continue
        flist = glob(in_path +dset +"_origin/*")
        new_flist = []
        ext_snts = []
        abs_snts = []
        all_abs_snts_len =[]
        n_cut = 0

        for fn in flist: #[:1000]:
            jd = json.load(open(fn,"r"))
            art = jd['article']
            abss = sepf(jd["abstract"][0])
            rnum = np.random.randint(10)
            #if len(abss) <30 or (len(abss) <40 and rnum >2) or (len(abss) <45 and rnum >6) or (len(abss) <50 and rnum >8) :
            if len(abss) <100 :
                new_ext = []
                if to_cut:              
                    absset = set(abss) - cwords
                    ext = [(ix,set(sepf(art[ix])) - cwords) for ix in jd['extracted']]
                    for ix, exset in ext:
                        interset = list(absset & exset)
                        if len(interset) / len(absset) > 0.1:
                            new_ext.append(ix)
                        else:
                            u_set = list(concat([xset for ixt, xset in ext if ixt!=ix]))
                            if len([w for w in interset if w not in u_set]) > k_uniq:
                                new_ext.append(ix)

                    #new_ext = [ix for ix, exset in ext if len(absset & exset) / len(absset) > 0.05]
                    if len(new_ext) < len(jd["extracted"]):
                        n_cut += 1
                        jd["extracted"] = new_ext
                new_flist.append(fn)
                ext_snts.append(' '.join([art[i] for i in jd["extracted"]]))
                abs_snts.append(abss)
                
                json.dump(jd, open(out_path+dset+"/"+fn.split('/')[-1],"w"), ensure_ascii=False, indent=4)
            """
            if len(ext[i_match[0]]) > 1.2*len(abss) and len(set(ext[i_match[0]]) & set(abss)) / len(set(abss)) >0.65:
                new_flist.append(fn)
                ext_snts.append(ext[i_match[0]])
                abs_snts.append(abss)
                jd["extracted"] = [jd["extracted"][i_match[0]]]
            """
                #json.dump(jd, open(out_path+fn.split('/')[-1],"w"), ensure_ascii=False,indent=4)
            all_abs_snts_len.append(len(abss))

        ext_snt_len = [len(sepf(s)) for s in ext_snts]
        abs_snt_len = [len(s) for s in abs_snts]   
        print(f"extracted sent_len  : mean = {np.mean(ext_snt_len)},  std = {np.std(ext_snt_len)}")
        print(f"abstracted sent_len : mean = {np.mean(abs_snt_len)},  std = {np.std(abs_snt_len)}")
        print(f"tot num of flist      : {len(flist)}")
        print(f"num of selected flist : {len(new_flist)}")
        print(f"all abst sent_len : mean = {np.mean(all_abs_snts_len)},  std = {np.std(all_abs_snts_len)}")
        print(f"num of cut flist : {n_cut}")

if __name__ == '__main__':
    make_new_fileset(to_shuffle=True, to_cut=False)
import nibabel as nib
import numpy as np
import glob
import os,sys
import difflib
import random
from subprocess import check_call

def grab_files(path,end):
    return glob.glob(os.path.join(path,end))


def split_cross_valid(slices_fn,segments_fn,hulls_fn):

    all_files =[]
    for n,seg_fn in enumerate(segments_fn):
        slice_nb = os.path.basename(seg_fn)[:4]
        slice_fn=difflib.get_close_matches(seg_fn,slices_fn)
        hull_fn=difflib.get_close_matches(seg_fn,hulls_fn)
        # slice_fn=[s for s in slices_fn if slice_base in s]
        if not slice_fn:
            print("Could not find an equivalent segment file {}".format(segment_fn))
            continue
        elif not hull_fn:
            print("Could not find an equivalent hull file {}".format(hull_fn))
            continue
        elif slice_nb not in hull_fn[0]:
            print("Could not find a hull file for {}".format(seg_fn))
            continue
        else:
            all_files.append((slice_fn[0],seg_fn,hull_fn[0]))

    return all_files


def get_files(fnDir):

    with open(fnDir) as f:
        # all_files are tuples with m and filenames
        # all_files = [tuple(i.split(' ')) for i in f]
        all_files = [tuple(i.strip()[1:-1].replace("'", "").split(', ')) for i in f]
    return all_files


test_fn = '/home/rpizarro/histo/weights/weights.brown.d2560/drop_000/XValidFns/test.txt'

files = get_files(test_fn)

# print(files)

slices_fn=[]
segments_fn=[]
slices=[]

for f in files:
    # ss=grab_files(p,'*')
    slices+=[os.path.basename(f[0])[:4]]
    slices_fn += [f[0]] #grab_files(p,'*/*jpg.nii')
    segments_fn += [f[1]] # grab_files(p,'*/*segmented.nii*')

# print(slices)

hull_path='/home/rpizarro/histo/data/rm311_65requad_hull'
hulls_fn=grab_files(hull_path,'*whole.hull.nii.gz')


# print(slices_fn)#,segments_fn,hulls_fn)

all_files = split_cross_valid(slices_fn,segments_fn,hulls_fn)

# print(all_files)

fn='/home/rpizarro/histo/data/txt_files/segmented_hull_files.txt'
thefile = open(fn,'w')
for item in all_files:
    thefile.write(' '.join(str(s) for s in item) + '\n')



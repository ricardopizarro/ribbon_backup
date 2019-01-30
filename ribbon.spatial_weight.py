import nibabel as nib
import numpy as np
import glob
import json
import random
import os
import sys
import difflib
from scipy import stats
from scipy import ndimage
from numpy import copy
from subprocess import check_call

# from keras.models import model_from_json
# from keras import backend as K
# from keras.callbacks import ModelCheckpoint, History
# from keras.optimizers import Adam
from keras.utils import np_utils


def get_channel(img):
    ch_ret=-1
    num_ch_labeled=0
    for ch in range(img.shape[2]):
        # print(ch)
        # print(np.unique(img[:,:,ch]))
        if len(np.unique(img[:,:,ch]))>2:
            ch_ret=ch
            num_ch_labeled+=1
    return ch_ret,num_ch_labeled


def consolidate_seg(seg):
    # swap elements labeled 6 and 2 to 0.  
    # elements labeled 6 indicate a tear in the white matter
    # elements labeled 5 indicate a fold in the gray matter
    # elements labeled 4 indicate subcortex or cerebellum
    # elements labeled 2 indicate a tear in the gray matter
    # elements labeled 3 indicate a blood vessel in the gray matter
    d={2:0,4:0,5:0,6:0,7:0,8:0,3:1}
    newArray = copy(seg)
    for k, v in d.items(): newArray[seg==k] = v
    # for m,row in enumerate(seg):
    #     for n,elem in enumerate(row):
    #         if elem in [3]:
    #             seg[m][n]=1
    #         elif elem in [2,4,5,6,7,8]:
    #             seg[m][n]=0
    return newArray


def resize_array_n_save(data,fn):

    print(data.shape)

    ratio=[1.0,0.1,0.1,1.0]
    # ratio=[0.5,0.5,1.0,1.0]
    data=ndimage.zoom(data,ratio,order=0)
    print(data.shape)
    # (1,X,Y,2) to (X,Y,2,1)
    data=np.rollaxis(data,0,4)
    print(data.shape)

    out_dir='/home/rpizarro/histo/data/attention_eg'
    
    fn_small=os.path.join(out_dir,fn)
    save_to_nii(data,fn_small)



def save_to_nii(data,fn):
    affine=np.eye(len(data.shape))
    img = nib.Nifti1Image(data,affine)
    path=fn
    print(path)
    if not os.path.isfile(path+'.gz'):
        nib.save(img,path)
        check_call(['gzip', path])
    else:
        print('File {} already exists'.format(path))


def prep_seg(seg_fn):
    seg_data = nib.load(segment_fn).get_data()

    ch,num_ch_labeled=get_channel(seg_data)
    if ch<0:
        print("{} does not have pink labels".format(segment_fn))
    elif num_ch_labeled>1:
        print("{} has too many channels with multiple labels".format(segment_fn))
    seg_data=seg_data[:,:,ch]

    seg_data=np.squeeze(consolidate_seg(seg_data))

    return seg_data


def get_sample_weights(seg,f):
    # seg is (1,X,Y,2)
    attention = np.zeros(seg.shape)
    # f = 10.0
    b = 0.05
    # seg[0,:,:,0] is (X,Y)
    D0 = ndimage.distance_transform_cdt(seg[0,:,:,0],metric='chessboard')
    D1 = ndimage.distance_transform_cdt(seg[0,:,:,1],metric='chessboard')
    D=D0+D1
    w = np.exp(1-D/f)
    w = 100*( w/np.max(w) + b )
    attention[0,:,:,0]=w
    attention[0,:,:,1]=w
    return attention


nb_slices=1

segment_fn = '/home/rpizarro/histo/data/rm311_65requad/0126_6x_concat_6x_whole.segmented.nii.gz'
seg_data = prep_seg(segment_fn)
seg_shape = seg_data.shape

# (X,Y)
print(seg_shape)
# (1,X,Y)
seg_data = seg_data.reshape((1,)+seg_shape)
# (X,Y,2)
seg_shape_categorical = seg_shape+(2,)
seg_data = np.reshape(np_utils.to_categorical(seg_data,2),seg_shape_categorical)

# (1,X,Y,2)
Y = seg_data.reshape((1,)+seg_shape_categorical)
# fn='0126_6x_concat_6x_whole.segmented.categorical.nii'
# resize_array_n_save(Y,fn)

for f in [20,50,80,120,150,200]:
    # (1,X,Y,2)
    Z = get_sample_weights(Y,f)
    fn='0126_6x_concat_6x_whole.spatial-weight.f{0:04d}.nii'.format(f)
    resize_array_n_save(Z,fn)





import os
import nibabel as nib
import numpy as np
import matplotlib
from PIL import Image

from keras.models import model_from_json, load_model
from keras.optimizers import Adam

# import cv2 as cv
import glob
import json
import sys
from scipy import ndimage,stats
from numpy import copy
import difflib
import random
from subprocess import check_call

matplotlib.use('Agg')
import matplotlib.pyplot as plt



def grab_files(path,end):
    return glob.glob(os.path.join(path,end))


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

    return newArray

def segment(tile,seg,hull,white):
    # We used the labeled seg to segment the subcortex and cerebellum
    # To mask this portion out we simply make it a high value of 10
    d={2:white,4:white,5:white,6:white,7:white,8:white}
    tmpArray = copy(tile)
    for k, v in d.items(): tmpArray[seg==k] = v

    d={0:white}
    newArray = copy(tmpArray)
    for k, v in d.items(): newArray[hull==k] = v

    return newArray


def swap_labels(tile,a,b):
    # we wish to swap elements in order to change the color in itksnap
    # this function swaps a for b
    for m,row in enumerate(tile):
        for n,elem in enumerate(row):
            if elem in [a]:
                tile[m][n]=b
    return tile

def normalize(tile):
    m=float(np.mean(tile))
    st=float(np.std(tile))
    if st > 0:
        norm = (tile - m) / float(st)
    else:
        norm = tile - m
    return norm


def get_channel(img):
    ch_ret=-1
    num_ch_labeled=0
    for ch in range(img.shape[2]):
        if len(np.unique(img[:,:,ch]))>1:
            ch_ret=ch
            num_ch_labeled+=1
    return ch_ret,num_ch_labeled

def rgb_2_lum(img):
    # the rgb channel is located at axis=2 for the data
    img=0.2126*img[:,:,0]+0.7152*img[:,:,1]+0.0722*img[:,:,2]
    return img


def avg_tile(slice_avg,single_tile,x,y,tile_width):
    slice_sum=slice_avg[0]
    slice_sum[x:x+tile_width,y:y+tile_width]+=single_tile

    slice_count=slice_avg[1]
    slice_count[x:x+tile_width,y:y+tile_width]+=1
    return slice_sum,slice_count


def retile(tiles,coord,slice_shape,tile_width):
    # slice_shape is rgb shape with a 3 at the end
    nb_tiles=tiles.shape[2]
    print(slice_shape)
    print(tiles.shape)
    print(coord)
    # typical size: (25,2666,2760)
    slice_sum=np.zeros(slice_shape)
    slice_count=np.zeros(slice_shape)
    slice_avg=[slice_sum,slice_count]
    # tabulate the elements here, we will do a final mode at the end
    slice = np.zeros(slice_shape)
    tidx=0
    for x,y in coord:
        single_tile=tiles[:,:,tidx,0]
        slice_avg=avg_tile(slice_avg,single_tile,x,y,tile_width)
        tidx+=1
    print('slice : ( {},{} )'.format(np.amin(slice_avg[0]),np.amax(slice_avg[0])))
    print('slice : {}'.format(np.unique(slice_avg[1])))
    slice=np.true_divide(slice_avg[0],slice_avg[1])
    # flip slice to make equivalent to original dataset
    # slice=slice[::-1,::-1]
    slice=np.reshape(slice,slice.shape+(1,1,))
    return slice


def get_new_model(verbose=False):
    fn = "../model/NN_brown_unet_d2560_c5p2.n2soft.model.json"
    with open(fn) as json_data:
        d = json.load(json_data)
    model = model_from_json(d)
    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=[jaccard_index])
    if verbose:
        print(model.summary())
    return model



def calc_jac_idx(data,str_arg='None'):
    data=data.astype(np.int64)
    try:
        y = np.bincount(data)
    except Exception as e:
        print("Warning: {}".format(e))
        print(data)
        print(str_arg)
        return 0.0
    ii = np.nonzero(y)[0]
    # tn = true negative, means the two raters agree it is not gray matter
    # fn = false negative, means jeeda marked it as not gray matter and deepthi marked as gray matter
    # fp = false positive, means jeeda marked it as gray matter and deepthi marked as not gray matter
    # tp = true positive,  means the two raters agree it is gray matter
    ii_str=['tn','fn','fp','tp']
    tn,fn,fp,tp = y
    print(zip(ii_str,y[ii]))
    return 1.0*tp/(tp+fp+fn)

def calc_dc(data,str_arg='None'):
    data=data.astype(np.int64)
    try:
        y = np.bincount(data)
    except Exception as e:
        print("Warning: {}".format(e))
        print(data)
        print(str_arg)
        return 0.0
    ii = np.nonzero(y)[0]
    # tn = true negative, means the two raters agree it is not gray matter
    # fn = false negative, means jeeda marked it as not gray matter and deepthi marked as gray matter
    # fp = false positive, means jeeda marked it as gray matter and deepthi marked as not gray matter
    # tp = true positive,  means the two raters agree it is gray matter
    ii_str=['tn','fn','fp','tp']
    tn,fn,fp,tp = y
    print(zip(ii_str,y[ii]))
    return 2.0*tp/(2*tp+fp+fn)

def save_to_nii(data,out_dir,fn):
    # out_dir='/home/rpizarro/histo/prediction/drop/20180522_hull/drop_030'
    affine=np.eye(len(data.shape))
    img = nib.Nifti1Image(data,affine)
    path=os.path.join(out_dir,fn)
    print(path)
    if not os.path.isfile(path+'.gz'):
        nib.save(img,path)
        check_call(['gzip', path])
    else:
        print('File {} already exists'.format(path))


def get_model(path,verbose=False):
    print(path)
    list_of_files = glob.glob(os.path.join(path,'model.set007*FINAL.h5'))
    if list_of_files:
        # print(list_of_files)
        model_fn = max(list_of_files, key=os.path.getctime)
        print('Loading model : {}'.format(model_fn))
        model = load_model(model_fn)
        if verbose:
            print(model.summary())
    else:
        print('We did not find any models.  Getting a new one!')
        model = get_new_model(verbose=verbose)
    return model




nissl_dir = '/home/rpizarro/histo/data/amir_grant/'
nissl_base = 'nissl_final_1.jpg'
nissl_fn = os.path.join(nissl_dir,nissl_base)


img = Image.open(nissl_fn)
# img.shape = (1644,1072)
img = img.convert(mode="L")
# 8618.7 = 5620/1072.0*1644
slice_shape = (8619,5620)
img = img.resize(slice_shape)
data = np.asarray( img, dtype="int32")
# np swaps dimensions to (5620, 8619)
print(data.shape)

fn='{0}.luminance.nii'.format(nissl_base)
save_to_nii(np.reshape(data,shape+(1,1,)),save_dir,fn)

# prepare for input
data=normalize(data)


save_dir = os.path.dirname('/home/rpizarro/histo/prediction/amir_grant/')
weights_dir = os.path.dirname("../weights/spatial/drop_{0:03d}/decay_{1:03d}/epochs_500/".format(int(100*drop),int(decay)))
model = get_model(weights_dir, verbose=True)

coord = [(0,0),(0,500),(0,1000),(0,1500),(0,2000),(0,2500),(0,2999)]

input_size=(2560,2560,1)
output_size=(2560,2560,2)
tile_width = 5620
nb_tiles=len(coord)

output_size_tiles=output_size[:-1]+(nb_tiles,)+(2,)
y_pred_tiles=np.zeros(output_size_tiles)
y_prob_tiles=np.zeros(output_size_tiles)

for tidx,(x,y) in enumerate(coord):
    print((tidx,x,y))
    tile=data[x:x+tile_width,y:y+tile_width]

    X_test = np.reshape(tiles,(1,)+input_size)
    y_prob = model.predict(X_test,verbose=1)
    y_pred = np.around(y_prob)
    print('y_prob : ( {} , {} )'.format(np.amin(y_prob),np.amax(y_prob)))
    print('y_pred : {}'.format(np.unique(y_pred)))


    y_prob_tiles[:,:,n,:]=y_prob[0]
    y_pred_tiles[:,:,n,:]=y_pred[0]



Y_pred_slice=retile(np.expand_dims(y_pred_tiles[:,:,:,0],axis=3),coord,slice_shape,tile_width)
Y_pred_slice=np.around(Y_pred_slice)
fn='{0}.prediction.000.{1:04d}tiled.nii'.format(nissl_base,nb_tiles)
save_to_nii(Y_pred_slice,save_dir,fn)

Y_prob_slice=retile(np.expand_dims(y_prob_tiles[:,:,:,1],axis=3),coord,slice_shape,tile_width)
Y_prob_slice=100*Y_prob_slice
fn='{0}.probability.001.{1:04d}tiled.nii'.format(nissl_base,nb_tiles)
save_to_nii(Y_prob_slice,save_dir,fn)

Y_pred_slice=retile(np.expand_dims(y_pred_tiles[:,:,:,1],axis=3),coord,slice_shape,tile_width)
Y_pred_slice=np.around(Y_pred_slice)
fn='{0}.prediction.001.{1:04d}tiled.nii'.format(nissl_base,nb_tiles)
save_to_nii(Y_pred_slice,save_dir,fn)



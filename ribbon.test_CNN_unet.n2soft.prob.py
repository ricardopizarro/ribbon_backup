import nibabel as nib
import numpy as np
import cv2 as cv
import glob
import json
import os,sys
import tensorflow as tf
from scipy import ndimage
import difflib
import random
from subprocess import check_call

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import model_from_json
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import np_utils

def grab_files(path,end):
    return glob.glob(os.path.join(path,end))

def get_coord_random(dim,tile_width,nb_tiles):
    # nx is the number of tiles in the x-direction to cover the edge
    nx=int(np.ceil(float(dim[0])/tile_width))
    # ny is the number of tiles in the y-direction to cover the edge
    ny=int(np.ceil(float(dim[1])/tile_width))
    gap=0 
    if nx>1:
        gap = (tile_width*nx-dim[0])/(nx-1)
    # uniformly sample along one dimension to cover the edge
    uni_x = [int(np.floor(i*(tile_width-gap))) for i in range(nx)]
    uni_x[-1]=dim[0]-tile_width
    edge_x=[0]*ny+[dim[0]-tile_width]*ny+uni_x*2
    x=list(np.random.random_integers(0,dim[0]-tile_width,nb_tiles))
    x=edge_x+x

    gap=0
    if ny>1:
        gap = (tile_width*ny-dim[1])/(ny-1)
    # uniformly sample along one dimension to cover the edge
    uni_y = [int(np.floor(i*(tile_width-gap))) for i in range(ny)]
    uni_y[-1]=dim[1]-tile_width
    edge_y=uni_y*2+[0]*nx+[dim[1]-tile_width]*nx
    y=list(np.random.random_integers(0,dim[1]-tile_width,nb_tiles))
    y=edge_y+y

    # nb_tiles = 2*int(np.ceil(float(dim)/tile_width))
    # gap = (tile_width*nb_tiles-dim)/(nb_tiles-1)
    # coord = [int(np.floor(i*(tile_width-gap))) for i in range(nb_tiles)]
    # coord[-1]=dim-tile_width
    return zip(x,y)


def consolidate_seg(seg):
    # swap elements labeled 6 and 2 to 0.  
    # elements labeled 6 indicate a tear in the white matter
    # elements labeled 5 indicate a fold in the gray matter
    # elements labeled 4 indicate subcortex or cerebellum
    # elements labeled 2 indicate a tear in the gray matter
    # elements labeled 3 indicate a blood vessel in the gray matter
    for m,row in enumerate(seg):
        for n,elem in enumerate(row):
            if elem in [3,5]:
                seg[m][n]=1
            elif elem in [2,4,6]:
                seg[m][n]=0
    return seg


def segment(tile,seg,hull):
    # We used the labeled seg to segment the subcortex and cerebellum
    # To mask this portion out we simply make it white like the background post normalization [10,10,10]
    for m,row in enumerate(seg):
        for n,elem in enumerate(row):
            # print('we aint got no hull')
            # print(hull[m][n])
            if elem in [4]:
                tile[m,n]=10
            elif 0 in hull[m][n]:
                tile[m,n]=20
    return tile


def swap_labels(tile,a,b):
    # we wish to swap elements in order to change the color in itksnap
    # this function swaps a for b
    for m,row in enumerate(tile):
        for n,elem in enumerate(row):
            if elem in [a]:
                tile[m][n]=b
    return tile

def normalize_tile(tile):
    m=float(np.mean(tile))
    st=float(np.std(tile))
    if st > 0:
        norm = (tile-m)/float(st)
    else:
        norm = tile - m
    return norm

def normalize(tile_rgb):
    # normalize by RGB channel
    tile_norm=np.zeros(tile_rgb.shape)
    for ch in range(tile_rgb.shape[2]):
	tile=tile_rgb[:,:,ch]
        tile_norm[:,:,ch]=normalize_tile(tile_rgb[:,:,ch])
    return tile_norm

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

def rgb_2_lum(img):
    # the rgb channel is located at axis=2 for the data
    img=0.2126*img[:,:,0]+0.7152*img[:,:,1]+0.0722*img[:,:,2]
    return img

def zero_pad(data,tile_width):
    shape=data.shape
    shape_min=(tile_width,tile_width)
    shape_target=tuple([max(i,j) for i,j in zip(shape,shape_min)])
    data_pad=np.zeros(shape_target)+255
    data_pad[:data.shape[0],:data.shape[1]]=data
    return data_pad


def gen_tiles(img_fn,seg_fn,hull_fn,tile_width,slice_nb,quad_nb,nb_tiles):
    img = nib.load(img_fn)
    data=rgb_2_lum(img.get_data())
    shape=data.shape
    print(shape)
    if any([s<tile_width for s in list(shape)]):
        data=zero_pad(data,tile_width)
        shape=data.shape
        print('changed shape to : {}'.format(shape))

    hull_data = nib.load(hull_fn).get_data()
    seg_data = nib.load(seg_fn).get_data()
    ch,num_ch_labeled=get_channel(seg_data)
    if ch<0:
        print("{} does not have pink labels".format(seg_fn))
    elif num_ch_labeled>1:
        print("{} has too many channels with multiple labels".format(seg_fn))

    coord=get_coord_random(shape,tile_width,nb_tiles)
    coord=sorted(list(set(coord)))
    print(coord)
    nb_tiles=len(coord)
    fn='{0}_6x_concat_6x_whole.jpg_{1}.orig.jpg.nii'.format(slice_nb,quad_nb,nb_tiles)
    save_to_nii(np.reshape(data,shape+(1,1,)),fn)

    # tiles should have dimension (20-30,560,560,3)
    tiles_norm = np.zeros([nb_tiles]+[tile_width]*2+[1])
    seg = np.zeros([nb_tiles]+[tile_width]*2)
    tidx=0
    for x,y in coord:
        print((tidx,x,y))
        seg_tile=seg_data[x:x+tile_width,y:y+tile_width,ch].tolist()
        hull_tile=hull_data[x:x+tile_width,y:y+tile_width,ch].tolist()
        tile_lum=data[x:x+tile_width,y:y+tile_width]
        tile_norm=normalize_tile(tile_lum)
        tiles_norm[tidx,:,:,0]=segment(tile_norm,seg_tile,hull_tile)

        seg_tmp=np.asarray(consolidate_seg(seg_tile))
        seg_pad=np.zeros(seg.shape[1:])
        try:
            seg_pad[:seg_tmp.shape[0],:seg_tmp.shape[1]]=seg_tmp
        except:
            print('Check the segmentation size: {}'.format(seg_tmp.shape))
        seg[tidx,:,:]=seg_pad

        tidx+=1
    return tiles_norm,seg,coord,shape


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
    # print(np.unique(slice_count))
    slice=np.true_divide(slice_avg[0],slice_avg[1])
    # flip slice to make equivalent to original dataset
    # slice=slice[::-1,::-1]
    slice=np.reshape(slice,slice.shape+(1,1,))
    return slice


def get_model(verbose=False):
    fn = "../model/NN_brown_unet_d2560_c5p2.n2soft.model.json"
    with open(fn) as json_data:
        d = json.load(json_data)
    model = model_from_json(d)
    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=[dice_coef])
    if verbose:
        print(model.summary())
    return model

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def calc_dc(data):
    data=data.astype(np.int64)
    y = np.bincount(data)
    ii = np.nonzero(y)[0]
    # tn = true negative, means the two raters agree it is not gray matter
    # fn = false negative, means jeeda marked it as not gray matter and deepthi marked as gray matter
    # fp = false positive, means jeeda marked it as gray matter and deepthi marked as not gray matter
    # tp = true positive,  means the two raters agree it is gray matter
    ii_str=['tn','fn','fp','tp']
    tn,fn,fp,tp = y
    print(zip(ii_str,y[ii]))
    return 2.0*tp/(2*tp+fp+fn)

def save_to_nii(data,fn):
    out_dir='/data/shmuel/shmuel1/rap/histo/prediction/subcortex/20180420_hull/'
    affine=np.eye(len(data.shape))
    img = nib.Nifti1Image(data,affine)
    path=os.path.join(out_dir,fn)
    print(path)
    nib.save(img,path)
    check_call(['gzip', path])


def testNN(slice_fn,segment_fn,hull_fn,nb_tiles_in,verbose=False):
    # nb_step is number of tiles per step
    input_size=(2560,2560,1)
    output_size=(2560,2560,2)
    batch_size=32
    tile_width=2560

    model = get_model(verbose=True)
    weights_fn='../weights/weights.brown.d2560/nohull_003/weights.set050.epochs5400.FINAL.h5'
    model.load_weights(weights_fn)

    slice_nb=os.path.basename(slice_fn)[:4]
    quad_nb =slice_fn.split('.jpg.')[0][-1]
    if verbose:
        print("{} : {}".format(slice_fn,segment_fn,hull_fn))
    # tiles_rgb are for viewing, tiles are normalized used for predicting
    # coord identifies the location of the tile,seg
    # slice_shape specifies the shape of the image
    nb_tiles=nb_tiles_in
    tiles_norm,seg,coord,slice_shape=gen_tiles(slice_fn,segment_fn,hull_fn,tile_width,slice_nb,quad_nb,nb_tiles)
    nb_tiles=len(coord)

    seg=np.reshape(np_utils.to_categorical(seg,output_size[-1]),(nb_tiles,)+output_size)

    output_size_tiles=output_size[:-1]+(nb_tiles,)+(2,)
    input_size_tiles=output_size[:-1]+(nb_tiles,)+(1,)

    y_true_tiles=np.zeros(output_size_tiles)
    y_prob_tiles=np.zeros(output_size_tiles)
    y_pred_tiles=np.zeros(output_size_tiles)
    dc_val=np.zeros(nb_tiles)

    for n in range(nb_tiles):
        X_test=np.reshape(tiles_norm[n],(1,)+input_size)
        y_true=np.reshape(seg[n],(1,)+output_size)
        y_prob = model.predict(X_test,batch_size=batch_size,verbose=0)
        y_pred = np.around(y_prob)
        y_pred=y_pred.astype(dtype=y_true.dtype)

        # getting the dice coefficient
        data12 = y_true[0]+2*y_pred[0]
        print(data12.shape)
        dc_man = calc_dc(data12.flatten())
        print('This is dice: {0:0.4f}'.format(dc_man))
        dc_val[n]=float(dc_man)

        y_true_tiles[:,:,n,:]=y_true[0]
        y_prob_tiles[:,:,n,:]=y_prob[0]
        y_pred_tiles[:,:,n,:]=y_pred[0]

    print('{} : {} : {}'.format(slice_nb,quad_nb,nb_tiles))

    Y_pred_slice=retile(np.expand_dims(y_pred_tiles[:,:,:,0],axis=3),coord,slice_shape,tile_width)
    Y_pred_slice=np.around(Y_pred_slice)
    fn='{0}_6x_concat_6x_whole.jpg_{1}.prediction.000.{2:04d}tiled.nii'.format(slice_nb,quad_nb,nb_tiles)
    save_to_nii(Y_pred_slice,fn)

    Y_prob_slice=retile(np.expand_dims(y_prob_tiles[:,:,:,1],axis=3),coord,slice_shape,tile_width)
    Y_prob_slice=100*Y_prob_slice
    fn='{0}_6x_concat_6x_whole.jpg_{1}.probability.001.{2:04d}tiled.nii'.format(slice_nb,quad_nb,nb_tiles)
    save_to_nii(Y_prob_slice,fn)

    Y_pred_slice=retile(np.expand_dims(y_pred_tiles[:,:,:,1],axis=3),coord,slice_shape,tile_width)
    Y_pred_slice=np.around(Y_pred_slice)
    fn='{0}_6x_concat_6x_whole.jpg_{1}.prediction.001.{2:04d}tiled.nii'.format(slice_nb,quad_nb,nb_tiles)
    save_to_nii(Y_pred_slice,fn)

    Y_true_slice=retile(np.expand_dims(y_true_tiles[:,:,:,1],axis=3),coord,slice_shape,tile_width)
    Y_true_slice=np.around(Y_true_slice)
    fn='{0}_6x_concat_6x_whole.jpg_{1}.truesegment.{2:04d}tiled.nii'.format(slice_nb,quad_nb,nb_tiles)
    save_to_nii(Y_true_slice,fn)

    data12_slice = Y_true_slice+2*Y_pred_slice
    dc_slice = calc_dc(data12_slice.flatten())
    print('Dice coefficient ( avg tiles | slice ) : ( {0:0.3f} | {0:0.3f} )'.format(np.mean(dc_val),dc_slice))

    bins = np.linspace(0, 1, 11)
    plt.hist(dc_val,bins)
    plt.title("Dice Coefficient Distribution")
    plt.xlabel("Dice Coefficient")
    plt.ylabel("Frequency")
    fn='{0}_6x_concat_6x_whole.jpg_{1}.DC_distribution.{2:04d}tiled.DC{3:0.3f}.png'.format(slice_nb,quad_nb,nb_tiles,np.mean(dc_val))
    plt.savefig('/data/shmuel/shmuel1/rap/histo/prediction/subcortex/20180420_hull/'+fn)
    plt.close()

    Y_out_slice=Y_true_slice+3*Y_pred_slice
    fn='{0}_6x_concat_6x_whole.jpg_{1}.segmented.{2:04d}tiled.DC{3:0.3f}.nii.gz'.format(slice_nb,quad_nb,nb_tiles,np.mean(dc_val))
    save_to_nii(Y_out_slice,fn)



slice_fn = sys.argv[1]
segment_fn = sys.argv[2]
hull_fn = sys.argv[3]

print('\n==Testing NN UNET ==\n')
nb_tiles=20
testNN(slice_fn,segment_fn,hull_fn,nb_tiles,verbose=True)




import nibabel as nib
import numpy as np
import cv2 as cv
import glob
import json
import os
import tensorflow as tf
from scipy import ndimage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import model_from_json
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

def grab_files(path,end):
    return glob.glob(path+end)

def get_coord_random(dim,tile_width,nb_tiles):
    # nx is the number of tiles in the x-direction to cover the edge
    nx=int(np.ceil(float(dim[0])/tile_width))
    # ny is the number of tiles in the y-direction to cover the edge
    ny=int(np.ceil(float(dim[1])/tile_width))

    gap = (tile_width*nx-dim[0])/(nx-1)
    # uniformly sample along one dimension to cover the edge
    uni_x = [int(np.floor(i*(tile_width-gap))) for i in range(nx)]
    uni_x[-1]=dim[0]-tile_width
    edge_x=[0]*ny+[dim[0]-tile_width]*ny+uni_x*2
    x=list(np.random.random_integers(0,dim[0]-tile_width,nb_tiles))
    x=edge_x+x

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
                tile[m,n,:]=10
            elif 0 in hull[m][n]:
                # print('we go some hull!')
                tile[m,n,:]=20
    return tile

def convexicate(data,hull):
    # We used the labeled hull to segment the brain from background
    # To mask this portion out we simply make it white like the background [255,255,255]
    for m,row in enumerate(hull):
        for n,elem in enumerate(row):
            if elem in [0]:
                data[m,n,:]=255
    return data


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


def gen_tiles(img_fn,seg_fn,hull_fn,tile_width,nb_tiles):
    img = nib.load(img_fn)
    hdr=img.header
    raw=hdr.structarr
    data = img.get_data()
    shape = img.shape

    hull_data = nib.load(hull_fn).get_data()
    # print(np.unique(hull_data))
    # hull_data = hull_data[:,:,1,0]
    # data = convexicate(data,hull_data.tolist())

    seg_data = nib.load(seg_fn).get_data()
    ch,num_ch_labeled=get_channel(seg_data)
    if ch<0:
        print("{} does not have pink labels".format(seg_fn))
    elif num_ch_labeled>1:
        print("{} has too many channels with multiple labels".format(seg_fn))
    # tile width will allow for three layers of 2x2 max pooling, divisible by 8=2*2*2
    coord=get_coord_random(shape,tile_width,nb_tiles)
    # print(coord)
    nb_tiles=len(coord)
    # tiles should have dimension (20-30,560,560,3)
    tiles = np.zeros([nb_tiles]+[tile_width]*2+[3])
    tiles_rgb = np.zeros([nb_tiles]+[tile_width]*2+[3])
    seg = np.zeros([nb_tiles]+[tile_width]*2)
    tidx=0
    for x,y in coord:
        # data=img.get_data()
        tile_tmp=data[x:x+tile_width,y:y+tile_width,:]
        tiles_rgb[tidx,:,:,:]=tile_tmp

        seg_tile=seg_data[x:x+tile_width,y:y+tile_width,ch].tolist()
        hull_tile=hull_data[x:x+tile_width,y:y+tile_width,ch].tolist()
        tile_norm=normalize(tile_tmp)
        tile_seg=segment(tile_norm,seg_tile,hull_tile)

        tiles[tidx,:,:,:]=tile_seg#normalize(tile_tmp)
        seg_tmp=np.asarray(consolidate_seg(seg_tile))
        seg_pad=np.zeros(seg.shape[1:])
        seg_pad[:seg_tmp.shape[0],:seg_tmp.shape[1]]=seg_tmp
        seg[tidx,:,:]=seg_pad
            
        tidx+=1
    return tiles_rgb,tiles,seg,coord,shape


def avg_tile(slice_avg,single_tile,x,y,tile_width):
    slice_sum=slice_avg[0]
    slice_sum[x:x+tile_width,y:y+tile_width]+=single_tile

    slice_count=slice_avg[1]
    slice_count[x:x+tile_width,y:y+tile_width]+=1
    return slice_sum,slice_count


def retile(tiles,coord,slice_shape,tile_width):
    # slice_shape is rgb shape with a 3 at the end
    nb_tiles=tiles.shape[2]
    # typical size: (25,2666,2760)
    slice_sum=np.zeros(slice_shape[:-1])
    slice_count=np.zeros(slice_shape[:-1])
    slice_avg=[slice_sum,slice_count]
    # tabulate the elements here, we will do a final mode at the end
    slice = np.zeros(slice_shape[:-1])
    tidx=0
    for x,y in coord:
        single_tile=tiles[:,:,tidx,0]
        slice_avg=avg_tile(slice_avg,single_tile,x,y,tile_width)
        tidx+=1
    # print(np.unique(slice_count))
    slice=np.true_divide(slice_avg[0],slice_avg[1])
    # flip slice to make equivalent to original dataset
    slice=slice[::-1,::-1]
    slice=np.reshape(slice,slice.shape+(1,1,))
    return slice


def get_edges(img):
    # print(img.shape[1:3])
    img=np.reshape(img,img.shape[1:3])
    sx = ndimage.sobel(img, axis=0, mode='constant')
    sy = ndimage.sobel(img, axis=1, mode='constant')
    sob = np.around(np.hypot(sx, sy))
    mag = np.max(sob)-np.min(sob)
    if mag > 1e-3:
        sob = np.around((sob-np.min(sob))/(mag))
    sob_dilated = ndimage.binary_dilation(sob).astype(sob.dtype)
    sob_dilated = ndimage.binary_dilation(sob_dilated).astype(sob.dtype)
    return sob,sob_dilated


def get_model(verbose=False):
    fn = "../model/NN_brown_unet.model.json"
    with open(fn) as json_data:
        d = json.load(json_data)
    model = model_from_json(d)
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    if verbose:
        print(model.summary())
    return model

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def save_to_nii(data,fn):
    out_dir='/data/shmuel/shmuel1/rap/histo/prediction/subcortex/20180125_hull/0885/'
    affine=np.eye(len(data.shape))
    img = nib.Nifti1Image(data,affine)
    path=out_dir+fn
    print(path)
    nib.save(img,path)

def bgr_2_gray(img):
    # the bgr channel is located at axis=2
    img=0.2989*img[:,:,2]+0.587*img[:,:,1]+0.114*img[:,:,0]
    return img

def rgb_2_gray(img):
    # the rgb channel is located at axis=3 for the tiles
    img=0.2989*img[:,:,:,0]+0.587*img[:,:,:,1]+0.114*img[:,:,:,2]
    return img

def annotate(tmp,dc):
    img=tmp
    # color is BGR, not RGB
    img_shape=img.shape
    img=np.swapaxes(img,0,1)
    img=img[::-1,::-1]
    print(img.shape)
    img=np.reshape(img,img.shape+(1,))
    img=np.repeat(img,[3],axis=2)
    print(img.shape)
    # img = cv.rectangle(img,(450,450),(500,500),(0,100,0),3)
    cv.putText(img,'{0:0.2f}'.format(dc),(100,100),cv.FONT_HERSHEY_PLAIN,3,(125,125,125),3,cv.LINE_AA)
    img=bgr_2_gray(img)
    img=np.reshape(img,img_shape)
    img=img[::-1,::-1]
    img=np.swapaxes(img,0,1)
    img=np.maximum(img,tmp)
    return img

def gen10tiles(slice_fn,tile_width):
    tmp=nib.load(slice_fn).get_data()
    # img=tmp
    # color is BGR, not RGB
    img_shape=tmp.shape
    img = np.zeros(img_shape,np.uint8)
    # m = tmp.transpose((1, 2, 0)).astype(np.uint8).copy() 
    # img=np.swapaxes(img,0,1)
    # img=img[::-1,::-1]
    # print(img.shape)
    # img.astype('float64')
    # print(img.dtype)
    nb_tiles=17
    x=list(np.random.random_integers(0,img_shape[1]-tile_width,nb_tiles))
    y=list(np.random.random_integers(0,img_shape[0]-tile_width,nb_tiles))
    for n in range(nb_tiles):
        top_left=(x[n],y[n])
        print(top_left)
        bottom_right=(x[n]+tile_width,y[n]+tile_width)
        print(bottom_right)
        cv.rectangle(img,top_left,bottom_right,(255,255,255),10)
    # img = m.transpose((2, 0, 1)).astype(tmp.dtype).copy()
    # img=np.maximum(img,tmp)
    print(img.shape)
    img=np.around(bgr_2_gray(img))
    print(img.shape)
    img=np.reshape(img,img_shape[:-1]+(1,1,))
    # img=img[::-1,::-1]
    # img=np.swapaxes(img,0,1)
    return img


def testNN(files,nb_tiles_in,verbose=False):
    # nb_step is number of tiles per step
    input_size=(560,560,3)
    output_size=(560,560,1)
    batch_size=32
    tile_width=560

    sess = tf.InteractiveSession()

    model = get_model(verbose=False)
    weights_fn='./weights.gpu.label1.brown/20180109.sub/weights.best.h5'
    model.load_weights(weights_fn)

    for l,line in enumerate(files):
        try:
            slice_fn=line[0]
            segment_fn=line[1]
            slice_nb=os.path.basename(segment_fn)[:4]
            hull_fn=os.path.join(os.path.dirname(slice_fn),'hull','{}-segment_hull.nii.gz'.format(slice_nb))
            if verbose:
                print("{} : {} : {}".format(slice_fn,segment_fn,hull_fn))
            # tiles_rgb are for viewing, tiles are normalized used for predicting
            # seg is segmentation done by deepthy
            # coord identifies the location of the tile,seg
            # slice_shape specifies the shape of the image
            nb_tiles=nb_tiles_in
            tiles_rgb,tiles,seg,coord,slice_shape=gen_tiles(slice_fn,segment_fn,hull_fn,tile_width,nb_tiles)
            nb_tiles=len(coord)
            seg=seg.reshape((nb_tiles,)+output_size)
            output_size_tiles=output_size[:-1]+(nb_tiles,)+(1,)
            X_test_tiles=np.reshape(np.rollaxis(rgb_2_gray(tiles_rgb),0,3),output_size_tiles)
            y_out_tiles=np.zeros(output_size_tiles)
            y_true_tiles=np.zeros(output_size_tiles)
            y_pred_tiles=np.zeros(output_size_tiles)
            dc_val=np.zeros(nb_tiles)

            X_test_slice=retile(X_test_tiles,coord,slice_shape,tile_width)
            fn='{0}-sliceindex_{1:04d}tiled.nii.gz'.format(slice_nb,nb_tiles)
            save_to_nii(X_test_slice,fn)

            X_test_10tiles=gen10tiles(slice_fn,tile_width)
            fn='{0}-sliceindex_{1:04d}tiled_10tiles.nii.gz'.format(slice_nb,nb_tiles)
            save_to_nii(X_test_10tiles,fn)

            for n in range(nb_tiles):
                X_test=np.reshape(tiles[n],(1,)+input_size)
                y_true=np.reshape(seg[n],(1,)+output_size)
                y_edge_thin,y_edge_thick=get_edges(y_true)
                y_pred = np.around(model.predict(X_test,batch_size=batch_size,verbose=0))
                y_pred=y_pred.astype(dtype=y_true.dtype)

                # getting the dice coefficient
                dc=dice_coef(y_true,y_pred)
                a=tf.Print(dc,[dc],message="This is a: ")
                b=a.eval()
                dc_val[n]=float(b)

                img = y_true[0,:,:,0]+3*y_pred[0,:,:,0]

                y_true_tiles[:,:,n,0]=np.reshape(y_true[0,:,:,0],output_size[:-1])
                y_pred_tiles[:,:,n,0]=np.reshape(y_pred[0,:,:,0],output_size[:-1])
                y_out_tiles[:,:,n,0] = np.reshape(img,output_size[:-1])


            fn='{0}-true_segment_{1:04d}tiles.nii.gz'.format(slice_nb,nb_tiles)
            save_to_nii(y_true_tiles,fn)
            fn='{0}-segmented_{1:04d}tiles.nii.gz'.format(slice_nb,nb_tiles)
            save_to_nii(y_out_tiles,fn)
            fn='{0}-sliceindex_{1:04d}tiles.nii.gz'.format(slice_nb,nb_tiles)
            save_to_nii(X_test_tiles,fn)

            Y_true_slice=retile(y_true_tiles,coord,slice_shape,tile_width)
            Y_true_slice=np.around(Y_true_slice)
            print(np.unique(Y_true_slice))
            fn='{0}-true_segment_{1:04d}tiled.nii.gz'.format(slice_nb,nb_tiles)
            save_to_nii(Y_true_slice,fn)

            Y_pred_slice=retile(y_pred_tiles,coord,slice_shape,tile_width)
            Y_pred_slice=np.around(Y_pred_slice)
            print(np.unique(Y_pred_slice))
            fn='{0}-pred_segment_{1:04d}tiled_dc{2:0.3f}.nii.gz'.format(slice_nb,nb_tiles,np.mean(dc_val))
            save_to_nii(Y_pred_slice,fn)
            
            Y_pred_thin,Y_pred_thick=get_edges(np.reshape(Y_pred_slice,(1,)+Y_pred_slice.shape[:-1]))
            Y_pred_thick=100*np.reshape(Y_pred_thick,Y_pred_slice.shape)
            fn='{0}-pred_edge_{1:04d}tiled_dc{2:0.3f}.nii.gz'.format(slice_nb,nb_tiles,np.mean(dc_val))
            save_to_nii(Y_pred_thick,fn)

            Y_true_edgePred=np.maximum(Y_true_slice,Y_pred_thick)
            fn='{0}-true_edgePred_{1:04d}tiled_dc{2:0.3f}.nii.gz'.format(slice_nb,nb_tiles,np.mean(dc_val))
            save_to_nii(Y_true_edgePred,fn)

            print('Dice coefficient average : {0:0.3f}'.format(np.mean(dc_val)))
            
            bins = np.linspace(0, 1, 11)
            plt.hist(dc_val,bins)
            plt.title("Dice Coefficient Distribution")
            plt.xlabel("Dice Coefficient")
            plt.ylabel("Frequency")
            fn='{0}-dice_distr_{1:04d}tiled_dc_avg{2:0.3f}.png'.format(slice_nb,nb_tiles,np.mean(dc_val))
            plt.savefig('/data/shmuel/shmuel1/rap/histo/prediction/subcortex/20180125_hull/0885/'+fn)
            plt.close()

            # Y_out_slice=retile(y_out_tiles,coord,slice_shape,tile_width)
            Y_out_slice=Y_true_slice+3*Y_pred_slice
            fn='{0}-segmented_{1:04d}tiled_dc{2:0.3f}.nii.gz'.format(slice_nb,nb_tiles,np.mean(dc_val))
            save_to_nii(Y_out_slice,fn)

        except Exception as e:
            print(str(e))
            pass

 
def split_train_valid(slices_fn,segments_fn):
    # print(sorted(slices_fn))
    train_files =[] 
    validation_files=[]
    for n,seg_fn in enumerate(segments_fn):
        slice_number = os.path.basename(seg_fn)[:4]
        slice_fn=[fn for fn in slices_fn if slice_number in fn]
        if not slice_fn:
            print("Could not find an equivalent slice for number {}".format(slice_number))
            continue
        # print(slice_fn)
        if np.mod(n,5):
            train_files.append((slice_fn[0],seg_fn))
        else:
            validation_files.append((slice_fn[0],seg_fn))
        # print(slice_number)
    # print(sorted(segments_fn))
    return train_files,validation_files



# data_path = "/data/shmuel/shmuel1/mok/histology_nhp/segmentation/HBP2/"
data_path = "/data/shmuel/shmuel1/rap/histo/data/"
slices_fn = grab_files(data_path,"*0885.jpg.nii.gz")
segments_fn = grab_files(data_path,"003_subcortex_cerebellum/0885-segmented.nii")
train_files,validation_files = split_train_valid(slices_fn,segments_fn)



print('\n==Testing NN UNET ==\n')
nb_tiles=10
# slices: 725,505,765,750,690
# index: 5,0,4,2,10
# runNN(all_train_files,all_valid_files,NN,nb_mods,nb_step,input_size)
# validation_files=train_files+validation_files
print(validation_files)
# valid_len=len(validation_files)
# validation_files=[validation_files[i] for i in range(13,valid_len)]
# testNN([validation_files[0]],nb_tiles,verbose=True)
testNN(validation_files,nb_tiles,verbose=True)



import nibabel as nib
import numpy as np
import cv2 as cv
import glob
import json
import os
import tensorflow as tf
from scipy import ndimage

from keras.models import model_from_json
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

def grab_files(path,end):
    return glob.glob(path+end)

def get_coord(dim,tile_width):
    num_tiles = int(np.ceil(dim/tile_width))
    gap = (tile_width*num_tiles-dim)/(num_tiles-1)
    coord = [int(np.floor(i*(tile_width-gap))) for i in range(num_tiles)]
    return coord


def consolidate_seg(seg):
    # swap elements labeled 6 and 2 to 0.  
    # elements labeled 6 indicate a tear in the white matter
    # elements labeled 4 indicate subcortex or cerebellum
    # elements labeled 2 indicate a tear in the gray matter
    # elements labeled 3 indicate a blood vessel in the gray matter
    for m,row in enumerate(seg):
        for n,elem in enumerate(row):
            if elem in [3]:
                seg[m][n]=1
            elif elem in [2,4,6]:
                seg[m][n]=0
    return seg


def segment(tile,seg):
    # We used the labeled seg to segment the subcortex and cerebellum
    # To mask this portion out we simply make it white like the background [255,255,255]
    for m,row in enumerate(seg):
        for n,elem in enumerate(row):
            if elem in [4]:
                tile[m,n,:]=255
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
    m=np.mean(tile)
    st=np.std(tile)
    if st > 0:
        norm = (tile - m) / st
    else:
        norm = (tile - m)
    return norm


def normalize(tile_rgb):
    # normalize by RGB channel
    for ch in range(tile_rgb.shape[2]):
	tile=tile_rgb[:,:,ch]
	tile_rgb[:,:,ch]=normalize_tile(tile)
    return tile_rgb

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


def gen_tiles(img_fn,seg_fn): # e.g. 
    img = nib.load(img_fn)
    hdr=img.header
    # print(hdr.get_xyzt_units())
    raw=hdr.structarr
    # print(raw['xyzt_units'])
    # print(img.get_data_dtype())
    data = img.get_data()
    shape = img.shape

    seg_data = nib.load(seg_fn).get_data()
    ch,num_ch_labeled=get_channel(seg_data)
    if ch<0:
        print("{} does not have pink labels".format(seg_fn))
    elif num_ch_labeled>1:
        print("{} has too many channels with multiple labels".format(seg_fn))
    # tile width will allow for three layers of 2x2 max pooling, divisible by 8=2*2*2
    tile_width = 560
    coord_x=get_coord(shape[0],tile_width)
    coord_y=get_coord(shape[1],tile_width)
    tiles = np.zeros([len(coord_x)*len(coord_y)]+[tile_width]*2+[3])
    tiles_rgb = np.zeros([len(coord_x)*len(coord_y)]+[tile_width]*2+[3])
    # print(tiles.shape)
    seg = np.zeros([len(coord_x)*len(coord_y)]+[tile_width]*2)
    # print(seg.shape)
    tidx=0
    for x in coord_x:
        for y in coord_y:
            # print(data[x:x+tile_width,y:y+tile_width,:].shape)
	    tile_tmp=data[x:x+tile_width,y:y+tile_width,:]
	    tile_pad=np.zeros(tiles.shape[1:])
	    tile_pad[:tile_tmp.shape[0],:tile_tmp.shape[1]]=tile_tmp
            tiles_rgb[tidx,:,:]=tile_pad

            seg_tile=seg_data[x:x+tile_width,y:y+tile_width,ch].tolist()
            tile_tmp=segment(tile_tmp,seg_tile)
	    tile_tmp=normalize(tile_tmp)
	    tile_pad[:tile_tmp.shape[0],:tile_tmp.shape[1]]=tile_tmp
            tiles[tidx,:,:,:]=tile_pad

            # channel 1 is where deepthy's labels are located
            seg_tmp=np.asarray(consolidate_seg(seg_tile))
	    seg_pad=np.zeros(seg.shape[1:])
	    seg_pad[:seg_tmp.shape[0],:seg_tmp.shape[1]]=seg_tmp
            seg[tidx,:,:]=seg_pad

            tidx+=1
    return tiles_rgb,tiles,seg


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
    return sob,sob_dilated

def randomize(tiles,seg):
    nb_slices=tiles.shape[0]
    random_perm = np.random.permutation(nb_slices)
    tiles=tiles[random_perm]
    seg=seg[random_perm]
    return tiles,seg


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
    out_dir='/data/shmuel/shmuel1/rap/histo/prediction/subcortex/'
    print(data.shape)
    affine=np.eye(len(data.shape))
    img = nib.Nifti1Image(data,affine)
    path=out_dir+fn
    print(path)
    nib.save(img,path)

def bgr_2_gray(img):
    img=0.2989*img[:,:,2]+0.587*img[:,:,1]+0.114*img[:,:,0]
    return img

def rgb_2_gray(img):
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
    print(img.dtype)
    # img = cv.rectangle(img,(450,450),(500,500),(0,100,0),3)
    cv.putText(img,'{0:0.2f}'.format(dc),(100,100),cv.FONT_HERSHEY_PLAIN,3,(125,125,125),3,cv.LINE_AA)
    print("STOP IT HERE!!!!")
    img=bgr_2_gray(img)
    img=np.reshape(img,img_shape)
    img=img[::-1,::-1]
    img=np.swapaxes(img,0,1)
    img=np.maximum(img,tmp)
    return img

def testNN(files,verbose=False):
    # nb_step is number of tiles per step
    input_size=(560,560,3)
    output_size=(560,560,1)
    batch_size=32

    sess = tf.InteractiveSession()

    model = get_model(verbose=False)
    weights_fn='./weights.gpu.label1.brown/20171221.sub/weights.best.h5'
    model.load_weights(weights_fn)

    for l,line in enumerate(files):
        try:
            slice_fn=line[0]
            segment_fn=line[1]
            slice_nb=os.path.basename(segment_fn)[:4]
            if verbose:
                print("{} : {}".format(slice_fn,segment_fn))
            # tiles_rgb are for viewing, tiles are normalized used for predicting
            # seg is segmentation done by deepthy
            tiles_rgb,tiles,seg=gen_tiles(slice_fn,segment_fn)
            # tiles,seg=randomize(tiles,seg)
            nb_tiles=tiles.shape[0]
            seg=seg.reshape((nb_tiles,)+output_size)
            output_size_tiles=output_size[:-1]+(nb_tiles,)+(1,)
            X_test_tiles=np.reshape(np.rollaxis(rgb_2_gray(tiles_rgb),0,3),output_size_tiles)
            print(X_test_tiles.shape)
            y_out_tiles=np.zeros(output_size_tiles)
            # X_test_tiles=np.zeros(output_size+(nb_tiles,))

            for n in range(nb_tiles):
                # print(tiles[n].shape)
                X_test=np.reshape(tiles[n],(1,)+input_size)
                # print(X_test.shape)
                y_true=np.reshape(seg[n],(1,)+output_size)
                
                y_edge_thin,y_edge_thick=get_edges(y_true)

                y_pred = np.around(model.predict(X_test,batch_size=batch_size,verbose=0))
                print(y_true.shape)
                print(y_pred.shape)
                y_pred=y_pred.astype(dtype=y_true.dtype)

                dc=dice_coef(y_true,y_pred)
                a=tf.Print(dc,[dc],message="This is a: ")
                # b=tf.add(a,a).eval()
                b=a.eval()
                # K.Print(dc)
                print(b)
                dc_val=float(b)

                y_out=np.zeros(input_size)
                y_out[:,:,0] = np.maximum(y_true[0,:,:,0],2*y_edge_thick)
                y_tmp = np.maximum(y_pred[0,:,:,0],2*y_edge_thin)
                # we update elements labeled 1 to be labeled 3
                y_out[:,:,1] = np.asarray(swap_labels(y_tmp.tolist(),1,3))
                y_out[:,:,2] = y_true[0,:,:,0]+3*y_pred[0,:,:,0]
                img=annotate(y_out[:,:,2],float(b))
                y_out_tiles[:,:,n,0] = np.reshape(img,output_size[:-1])

                fn='tiles/{0}-segmented_{1}of{2}.pred.dc{3:0.2f}.nii.gz'.format(slice_nb,n+1,nb_tiles,float(b))
                save_to_nii(np.reshape(y_out,input_size+(1,)),fn)
                # fn='{}-segmented_{}of{}.true.nii.gz'.format(slice_nb,n+1,nb_tiles)
                # save_to_nii(np.reshape(y_true,output_size+(1,)),fn)
                fn='tiles/{}-sliceindex_{}of{}.tile.nii.gz'.format(slice_nb,n+1,nb_tiles)
                save_to_nii(np.reshape(X_test,input_size+(1,)),fn)
            fn='{0}-segmented_all_tiles.pred.nii.gz'.format(slice_nb)
            save_to_nii(y_out_tiles,fn)
            fn='{}-sliceindex_all_tiles.nii.gz'.format(slice_nb)
            save_to_nii(X_test_tiles,fn)


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
slices_fn = grab_files(data_path,"*.jpg.nii.gz")
segments_fn = grab_files(data_path,"003_subcortex_cerebellum/*segmented.nii")
train_files,validation_files = split_train_valid(slices_fn,segments_fn)

print('\n==Testing NN UNET ==\n')
# runNN(all_train_files,all_valid_files,NN,nb_mods,nb_step,input_size)
testNN(validation_files[:5],verbose=True)





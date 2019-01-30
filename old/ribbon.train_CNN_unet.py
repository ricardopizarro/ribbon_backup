import nibabel as nib
import numpy as np
import glob
import json
import os

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

def get_coord_random(dim,tile_width,nb_tiles):

    return list(np.random.random_integers(0,dim-tile_width,nb_tiles))

def consolidate_seg(seg):
    # swap elements labeled 6 and 2 to 0.  
    # elements labeled 6 indicate a tear in the white matter
    # elements labeled 2 indicate a tear in the gray matter
    # elements labeled 3 indicate a blood vessel in the gray matter
    for m,row in enumerate(seg):
        for n,elem in enumerate(row):
            if elem in [3]:
		seg[m][n]=1 # make blood vessels as gray matter
            elif elem in [2,6]:
                seg[m][n]=0
    return seg

def normalize_tile(tile):
    m=np.mean(tile)
    st=np.std(tile)
    norm = (tile - m) / st
    return norm

def normalize_range(tile):
    if np.minimum(tile)<0:
        print('This tile has a minimum less than zero!')
    elif np.maximum(tile)>255:
        print('This tile has a maximum greater than 255!')
    return tile 


def normalize(tile_rgb):
    # normalize by RGB channel
    for ch in range(tile_rgb.shape[2]):
	tile=tile_rgb[:,:,ch]
	tile_rgb[:,:,ch]=normalize_tile(tile)
	# tile_rgb[:,:,ch]=normalize_range(tile)
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


def gen_tiles_random(img_fn,seg_fn,nb_tiles=5):
    img = nib.load(img_fn)
    data = img.get_data()
    shape = img.shape

    # print('\n{}'.format(img_fn))
    # print('{}'.format(seg_fn))
    seg_data = nib.load(seg_fn).get_data()
    ch,num_ch_labeled=get_channel(seg_data)
    if ch<0:
        print("{} does not have pink labels".format(seg_fn))
    elif num_ch_labeled>1:
        print("{} has too many channels with multiple labels".format(seg_fn))
#     else:
#         print('{} is alllllllll good, channel:{}'.format(seg_fn,ch))
    # tile width will allow for three layers of 2x2 max pooling, divisible by 8=2*2*2
    tile_width = 560
    coord_x=get_coord_random(shape[0],tile_width,nb_tiles)
    coord_y=get_coord_random(shape[1],tile_width,nb_tiles)
    tiles = np.zeros([nb_tiles]+[tile_width]*2+[3])
    # print(tiles.shape)
    seg = np.zeros([nb_tiles]+[tile_width]*2)
    # print(seg.shape)
    tidx=0
    for tidx in range(nb_tiles):
        x = coord_x[tidx]
        y = coord_y[tidx]
        # print(data[x:x+tile_width,y:y+tile_width,:].shape)
	tile_tmp=data[x:x+tile_width,y:y+tile_width,:]
	tile_tmp=normalize(tile_tmp)
	tile_pad=np.zeros(tiles.shape[1:])
	tile_pad[:tile_tmp.shape[0],:tile_tmp.shape[1]]=tile_tmp
        tiles[tidx,:,:,:]=tile_pad

        # channel ch determined above specifies Deepthy's labels
        seg_tmp=np.asarray(consolidate_seg(seg_data[x:x+tile_width,y:y+tile_width,ch].tolist()))
        seg_pad=np.zeros(seg.shape[1:])
        try:
            seg_pad[:seg_tmp.shape[0],:seg_tmp.shape[1]]=seg_tmp
        except:
            print('Check the segmentation size: {}'.format(seg_tmp.shape))

        seg[tidx,:,:]=seg_pad

    return tiles,seg



def gen_tiles(img_fn,seg_fn): # e.g. 
    img = nib.load(img_fn)
    data = img.get_data()
    shape = img.shape

    # print('\n{}'.format(img_fn))
    # print('{}'.format(seg_fn))
    seg_data = nib.load(seg_fn).get_data()
    ch,num_ch_labeled=get_channel(seg_data)
    if ch<0:
        print("{} does not have pink labels".format(seg_fn))
    elif num_ch_labeled>1:
        print("{} has too many channels with multiple labels".format(seg_fn))
#     else:
#         print('{} is alllllllll good, channel:{}'.format(seg_fn,ch))
    # tile width will allow for three layers of 2x2 max pooling, divisible by 8=2*2*2
    tile_width = 560
    coord_x=get_coord(shape[0],tile_width)
    coord_y=get_coord(shape[1],tile_width)
    tiles = np.zeros([len(coord_x)*len(coord_y)]+[tile_width]*2+[3])
    # print(tiles.shape)
    seg = np.zeros([len(coord_x)*len(coord_y)]+[tile_width]*2)
    # print(seg.shape)
    tidx=0
    for x in coord_x:
        for y in coord_y:
            # print(data[x:x+tile_width,y:y+tile_width,:].shape)
	    tile_tmp=data[x:x+tile_width,y:y+tile_width,:]
	    tile_tmp=normalize(tile_tmp)
	    tile_pad=np.zeros(tiles.shape[1:])
	    tile_pad[:tile_tmp.shape[0],:tile_tmp.shape[1]]=tile_tmp
            tiles[tidx,:,:,:]=tile_pad

            # channel 1 specifies Deepthy's labels
            seg_tmp=np.asarray(consolidate_seg(seg_data[x:x+tile_width,y:y+tile_width,1].tolist()))
	    seg_pad=np.zeros(seg.shape[1:])
            try:
                seg_pad[:seg_tmp.shape[0],:seg_tmp.shape[1]]=seg_tmp
            except:
                print(seg_tmp.shape)
                print(seg_pad.shape)
                print(img.shape)
                print(seg_data.shape)
                print(coord_x)
                print(x)
                print(x+tile_width)
                print(seg_data[x:x+tile_width,y:y+tile_width,1].shape)
                print(seg_data[x:x+tile_width,y:y+tile_width,1].tolist())
                print(consolidate_seg(seg_data[x:x+tile_width,y:y+tile_width,1].tolist()))
                seg_pad=seg_tmp[:seg_pad.shape[0],:seg_pad.shape[1]]

            seg[tidx,:,:]=seg_pad

            tidx+=1
    return tiles,seg


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


def runNN(train_files,valid_files):
    # nb_step is number of tiles per step
    input_size=(560,560,3)
    output_size=(560,560,1)
    # number of tiles per step
    nb_step=50 #20
    model = get_model(verbose=False)
    # track performance (dice coefficient loss) on train and validation datasets
    checkpath="./weights.gpu.label1.brown/weights.nb_step_050.{epoch:04d}_of_1000_dice_coef_loss_{loss:0.2f}.h5"
    checkpointer=ModelCheckpoint(checkpath, monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto')
    # fit the model using the data generator defined below
    model.fit_generator(fileGenerator(train_files,nb_step=nb_step,verbose=False,input_size=input_size,output_size=output_size), steps_per_epoch=4, epochs=1000, verbose=1,
            validation_data=fileGenerator(valid_files,nb_step=1,verbose=False,input_size=input_size,output_size=output_size),validation_steps=1,callbacks=[checkpointer])
    # save the weights at the end of epochs
    model.save_weights('./weights.gpu.label1.brown/weights.nb_step_050.1000ep.FINAL.h5',overwrite=True)

def fileGenerator(files,nb_step=5,verbose=True,input_size=(560,560,3),output_size=(560,560,1)):
    X = np.zeros((nb_step,) + input_size )
    Y = np.zeros((nb_step,) + output_size )
    n = 0
    while True:
        while n < nb_step:
            try:
                i = np.random.randint(0,len(files))
		# print(i)
                slice_fn=files[i][0]
                segment_fn=files[i][1]
                if verbose:
                    print("{} : {}".format(slice_fn,segment_fn))
                # tiles,seg=gen_tiles(slice_fn,segment_fn)
                tiles,seg=gen_tiles_random(slice_fn,segment_fn,nb_step)
                # tiles,seg=randomize(tiles,seg)
                nb_slices=tiles.shape[0]
                seg=seg.reshape((nb_slices,)+output_size)
                if nb_step-n < nb_slices:
                    need_slices=nb_step-n
                    X[n:n+need_slices]=tiles[:need_slices]
                    Y[n:n+need_slices]=seg[:need_slices]
                else:
                    X[n:n+nb_slices]=tiles
                    Y[n:n+nb_slices]=seg
            	    # print(X.shape)
                n+=nb_slices
            except Exception as e:
                print(str(e))
                pass
	if X.size:
            yield X,Y
	else:
	    print("X is empty!!!")
	    continue


def split_train_valid(slices_fn,segments_fn):
    # Need to edit this so that you have a testing dataset as well
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



# data_path = "/data/shmuel/shmuel1/mok/histology_nhp/segmentation/HBP2B_Corrections/"
data_path = "/data/shmuel/shmuel1/rap/histo/data/"
slices_fn = grab_files(data_path,"*.jpg.nii.gz")
segments_fn = grab_files(data_path,"*segmented.nii")
train_files,validation_files = split_train_valid(slices_fn,segments_fn)

'''
for line in train_files+validation_files:
    slice_fn=line[0]
    segment_fn=line[1]
    tiles,seg=gen_tiles_random(slice_fn,segment_fn)

'''
print('\n==Training NN UNET ==\n')
# runNN(all_train_files,all_valid_files,NN,nb_mods,nb_step,input_size)
runNN(train_files,validation_files)



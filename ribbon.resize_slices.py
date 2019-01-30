import nibabel as nib
import numpy as np
import glob
import sys
import os
import difflib
import scipy.ndimage
from subprocess import check_call

def grab_files(path,end):
    return glob.glob(os.path.join(path,end))



def resize_file(fn):

    data=nib.load(fn).get_data()
    print(data.shape)

    ratio=[0.1,0.1,1.0,1.0]
    # ratio=[0.5,0.5,1.0,1.0]
    data=scipy.ndimage.zoom(data,ratio,order=0)
    print(data.shape)
    # data_rgb = np.repeat(data,3,axis=2)
    # one time hack for slice 0096
    # data=data[::-1,::-1,...]
    if len(data.shape)<4:
        ndim=len(data.shape)
        data=np.reshape(data,data.shape+(4-ndim)*(1,))

    in_dir,basename=os.path.split(fn)
    out_dir=os.path.join(in_dir,'resize')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fn_small=os.path.join(out_dir,basename.replace('.gz',''))
    save_to_nii(data,fn_small)
    # fn_rgb=os.path.join(out_dir,'rgb_'+basename.replace('.gz',''))
    # save_to_nii(data_rgb,fn_rgb)



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


# for d in ['030','030_thresh','050','070']:
# data_path='/d1/deepthi/RM311_HighRes_Seg_Set1_1-74/'
data_path='/home/rpizarro/histo/prediction/spatial_weight/drop_030/decay_000/set198/'
# data_path=os.path.join(root_path,'drop_{}'.format(d))

if not os.access(data_path, os.R_OK):
    print('Cannot read any of the files in {}'.format(data_path))
    sys.exit()

files=grab_files(data_path,"*.nii.gz")

if not files:
    print('We did not find any files in {}'.format(data_path))
else:
    for f in files:
        print(f)
        resize_file(f)



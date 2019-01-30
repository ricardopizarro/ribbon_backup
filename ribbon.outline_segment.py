import nibabel as nib
import numpy as np
import os
from scipy import ndimage
from subprocess import check_call


def outline_segment_file(seg_fn):

    data=nib.load(seg_fn).get_data()
    print(data.shape)

    if len(data.shape)<4:
        ndim=len(data.shape)
        data=np.reshape(data,data.shape+(4-ndim)*(1,))

    sx = ndimage.sobel(data, axis=0, model='constant')
    sy = ndimage.sobel(data, axis=1, model='constant')
    data_outline = np.hypot(sx, sy)

    in_dir,basename=os.path.split(fn)
    out_dir=in_dir
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fn_segment=os.path.join(out_dir,basename.replace('.nii.gz','.outline.nii'))
    save_to_nii(data_outline,fn_segment)


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



seg_fn='/home/rpizarro/histo/prediction/spatial_weight/drop_030/decay_000/set102/0094_6x_concat_6x_whole.jpg.truesegment.0134tiled.nii.gz'

outline_segment_file(seg_fn)



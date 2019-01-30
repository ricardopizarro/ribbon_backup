import mahotas
# Citation:Coelho, L.P. 2013. Mahotas: Open source software for scriptable computer vision. Journal of Open Research Software 1(1):e3, DOI: http://dx.doi.org/10.5334/jors.ac
import nibabel as nib
import numpy as np
from numpy import copy
# import cv2 as cv
import glob
import os,sys
from scipy import ndimage
import difflib
from subprocess import check_call

def consolidate(seg):
    # swap elements labeled 6 and 2 to 0.  
    # elements labeled 6 indicate a tear in the white matter
    # elements labeled 5 indicate a fold in the gray matter
    # elements labeled 4 indicate subcortex or cerebellum
    # elements labeled 2 indicate a tear in the gray matter
    # elements labeled 3 indicate a blood vessel in the gray matter
    d={2:1,4:1,5:1,6:1,7:1,8:1,3:1}
    newArray = copy(seg)
    for k, v in d.items(): newArray[seg==k] = v

    # for m,row in enumerate(seg):
    #     for n,elem in enumerate(row):
    #         if elem in range(2,7):
    #             seg[m][n]=1
    return newArray


def segment(tile,seg):
    # We used the labeled seg to segment the subcortex and cerebellum
    # To mask this portion out we simply make it white like the background [255,255,255]
    for m,row in enumerate(seg):
        for n,elem in enumerate(row):
            if elem in [4]:
                tile[m,n,:]=10
    return tile

def swap_labels(tile,a,b):
    # we wish to swap elements in order to change the color in itksnap
    # this function swaps a for b
    for m,row in enumerate(tile):
        for n,elem in enumerate(row):
            if elem in [a]:
                tile[m][n]=b
    return tile


def get_channel(img):
    ch_ret=-1
    num_ch_labeled=0
    for ch in range(img.shape[2]):
        # print(ch)
        # print(np.unique(img[:,:,ch]))
        if len(np.unique(img[:,:,ch]))>1:
            ch_ret=ch
            num_ch_labeled+=1
    return ch_ret,num_ch_labeled




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


def convex_hull_segment(fdata):
    # Takes the range of values in the volume and creates a subvolume and calls a convexhull fill method
    # This uses mahatos based fill_convexhull method 
    # First two x slices are returning garbage values so removing those slices from the volume.

    # finput_file=plt.imread(finput_volume)
    # finput_file = rgb2gray(finput_file) 
    # fdata=finput_file

    zeros=np.zeros((fdata.shape[0],fdata.shape[1]))
    points=np.array(np.where(fdata[:,:]>0.0))
    if len(points)>=0:
        returnedCanvas=mahotas.polygon.fill_convexhull((fdata[:,:]>0.0)*1)
        zeros[2:-2,2:-2]=returnedCanvas[2:-2,2:-2]
    	# plt.imshow(zeros)
	# plt.show()
    else:
        print("No points in this slice to be hulled.")

    # Saving data as convex_hull.nii.gz
    #img = nib.Nifti1Image(zeros, finput_file.affine) # Creating array of zeros with the shape of final_brain_volume_mask
    #img.to_filename(os.path.join(fprefix,outputfile))
    return zeros



def convexicate(slice_fn,segment_fn,verbose=False):
    out_dir='/home/rpizarro/histo/data/rm311_128requad_test_hull/'
    slice_nb=os.path.basename(segment_fn)[:4]
    if verbose:
        print("{} : {}".format(slice_fn,segment_fn))
    segment_data = nib.load(segment_fn).get_data()
    ch,num_ch_labeled=get_channel(segment_data)
    if ch<0:
        print("{} does not have pink labels".format(segment_fn))
    elif num_ch_labeled>1:
        print("{} has too many channels with multiple labels".format(segment_fn))
    segment_data=np.squeeze(segment_data[:,:,ch])
    print(segment_data.shape)
    segment_data = consolidate(segment_data)
    segment_hull = convex_hull_segment(segment_data)
    segment_zeros= np.zeros(segment_hull.shape+(3,1))
    print(segment_zeros.shape)
    segment_zeros[:,:,1,0]=segment_hull
    segment_hull = np.reshape(segment_zeros,segment_hull.shape+(3,1,))
    # segment_hull = np.repeat(segment_hull,3,axis=2)
    fn='{0}_6x_concat_6x_whole.single_hull.nii'.format(slice_nb)
    fn_path=os.path.join(out_dir,fn)
    save_to_nii(segment_hull,fn_path)

def get_files(fnDir):
    with open(fnDir) as f:
        all_files = [eval(i) for i in f]
    return all_files


# Book keeping
print("Executing:",__file__)
print("Contents of the file during execution:\n",open(__file__,'r').read())

test_fn = '/home/rpizarro/histo/weights/weights.spatial.weight/drop_030/decay_000/XValidFns/test.txt'


files = get_files(test_fn)

# slice_fn = sys.argv[1]
# segment_fn = sys.argv[2]


print('\n==Convexicating the file ==\n')
# print(files)
for slice_fn, segment_fn in files[13:]:
    convexicate(slice_fn,segment_fn,verbose=True)

# filename=sys.argv[1]
# ConvexHullCerebellum(fprefix=os.getcwd(),finput_volume=filename,outputfile='test_hull.png')


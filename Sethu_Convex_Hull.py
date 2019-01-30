import mahotas
# Citation:Coelho, L.P. 2013. Mahotas: Open source software for scriptable computer vision. Journal of Open Research Software 1(1):e3, DOI: http://dx.doi.org/10.5334/jors.ac
import numpy as np
import nibabel as nib
import subprocess as sp
import sys
import matplotlib.pyplot as plt
import os


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def ConvexHullCerebellum(fprefix,finput_volume,outputfile):
	# Takes the range of values in the volume and creates a subvolume and calls a convexhull fill method
	# This uses mahatos based fill_convexhull method 
	# First two x slices are returning garbage values so removing those slices from the volume.

	finput_file=plt.imread(finput_volume)
	finput_file = rgb2gray(finput_file) 
	fdata=finput_file
	print(fdata.shape)

	zeros=np.zeros((fdata.shape[0],fdata.shape[1]))
	points=np.array(np.where(fdata[:,:]>0.0))
	if len(points)>=0:

		returnedCanvas=mahotas.polygon.fill_convexhull((fdata[:,:]>0.0)*1)
		zeros[2:-2,2:-2]=returnedCanvas[2:-2,2:-2]
		plt.imshow(zeros)
		plt.show()
	else:
		print("No points in this slice to be hulled.")

	# Saving data as convex_hull.nii.gz
	#img = nib.Nifti1Image(zeros, finput_file.affine) # Creating array of zeros with the shape of final_brain_volume_mask
	#img.to_filename(os.path.join(fprefix,outputfile))


filename=sys.argv[1]
ConvexHullCerebellum(fprefix=os.getcwd(),finput_volume=filename,outputfile='test_hull.png')

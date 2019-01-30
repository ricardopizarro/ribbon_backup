import numpy as np
import glob
import sys
import os
from PIL import Image
from fpdf import FPDF

def grab_files(path,end):
    return glob.glob(os.path.join(path,end))

def get_jac_idx(segment_fn):
    if 'jac_idx' in segment_fn:
        fn_parts=segment_fn.split('jac_idx')
        return fn_parts[1][:5]
    else:
        return ''


for d in ['070']:
    data_path='/home/rpizarro/histo/prediction/spatial_weight/drop_030/decay_000/set198/resize/pdf'

    for img_type in ['overlay','slice','segment']:

        png_files = sorted(grab_files(data_path,img_type+'/*.png'))

        pdf = FPDF()
        # imagelist is the list with all image filenames
        for image in png_files:
            print(image)
            slice_nb=os.path.basename(image)[:4]
            im = Image.open(image)
            if 'overlay' in img_type: 
                jac_idx=get_jac_idx(image)
            txt_corner='{} : {}'.format(slice_nb,jac_idx)
            w0,h0 = im.size
            w=210.0
            h=(w/w0)*h0
            pdf.add_page()
            pdf.image(image,0,0,w,h)
            pdf.set_xy(0, 0)
            pdf.set_font('arial', 'B', 13.0)
            pdf.cell(ln=0, h=5.0, align='L', w=0, txt=txt_corner, border=0)
        fn=os.path.join(data_path,img_type+'.pdf')
        pdf.output(fn, "F")






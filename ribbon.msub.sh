#!/bin/sh

slice_nb_file='/home/rpizarro/histo/data/txt_files/slice_nb_seg.two.txt'
while IFS=" " read -r slice_nb remainder; do
    msub ribbon.HELIOS_GPU.requad.sh $slice_nb
done <$slice_nb_file




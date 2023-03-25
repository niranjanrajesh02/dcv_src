#! /bin/bash
#PBS -N dcv_p1_edge
#PBS -o out.log
#PBS -e err.log
#PBS -l ncpus=50
#PBS -q gpu

rm /home/niranjan.rajesh_ug23/DCV/dcv_src/P1_Edges/out.log
rm /home/niranjan.rajesh_ug23/DCV/dcv_src/P1_Edges/err.log
rm -r /home/niranjan.rajesh_ug23/DCV/dcv_src/P1_Edges/Results
mkdir /home/niranjan.rajesh_ug23/DCV/dcv_src/P1_Edges/Results
source /apps/compilers/anaconda3/etc/profile.d/conda.sh
conda activate computer-vision
python /home/niranjan.rajesh_ug23/DCV/dcv_src/P1_Edges/main.py

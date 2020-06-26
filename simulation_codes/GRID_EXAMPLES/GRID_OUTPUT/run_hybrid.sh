#!/bin/bash
#
#PBS -l select=1:ncpus=1:mem=2GB
#PBS -l walltime=12:00:00           
#PBS -k oe
#PBS -q testq
#PBS -m bae
#PBS -M joshua.s.williams@uon.edu.au
source /etc/profile.d/modules.sh
module load numba/0.49.1-python.3.6
cd $PBS_O_WORKDIR
python /home/c3134027/PREDCORR_1D_INJECT_OPT/main_1D.py
exit 0

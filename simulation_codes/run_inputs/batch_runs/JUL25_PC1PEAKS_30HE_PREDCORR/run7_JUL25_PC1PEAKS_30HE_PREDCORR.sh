#!/bin/bash
#PBS -l select=1:ncpus=16:mem=10GB
#PBS -l walltime=100:00:00 
#PBS -k oe
#PBS -m ae
#PBS -M joshua.s.williams@uon.edu.au
# Autocreated by Python
source /etc/profile.d/modules.sh
module load numba/0.49.1-python.3.6
cd $PBS_O_WORKDIR
python /home/c3134027/hybrid/simulation_codes/PREDCORR_1D_PARALLEL/main_1D.py -r /batch_runs/JUL25_PC1PEAKS_30HE_PREDCORR/run_params.run -p /batch_runs/JUL25_PC1PEAKS_30HE_PREDCORR/plasma_params_20130725_213320_JUL25_PC1PEAKS_30HE_PREDCORR.plasma -n 7
exit 0

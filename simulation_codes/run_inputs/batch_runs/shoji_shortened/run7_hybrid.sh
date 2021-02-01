#!/bin/bash
# Created 24/06/2020 by c3134027
#PBS -l select=1:ncpus=16:mem=15GB
#PBS -l walltime=100:00:00           
#PBS -k oe
#PBS -m ae
#PBS -M joshua.s.williams@uon.edu.au
source /etc/profile.d/modules.sh
module load numba/0.49.1-python.3.6
cd $PBS_O_WORKDIR
python /home/c3134027/hybrid/simulation_codes/PREDCORR_1D_PARALLEL/main_1D.py -r /batch_runs/shoji_shortened/run_params_shoji2013.run  -p /batch_runs/shoji_shortened/plasma_params_7.plasma -n 7
exit 0

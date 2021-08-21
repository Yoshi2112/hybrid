#!/bin/bash
# Created 24/06/2020 by c3134027
#PBS -l select=1:ncpus=32:mem=5GB
#PBS -l walltime=150:00:00           
#PBS -k oe
#PBS -m ae
#PBS -M joshua.s.williams@uon.edu.au
source /etc/profile.d/modules.sh
module load numba/0.49.1-python.3.6
cd $PBS_O_WORKDIR
python /home/c3134027/hybrid/simulation_codes/CAM_CL_1D_PARALLEL/main_1D.py -r /batch_runs/new_shoji_full/run_params_shoji2013.run  -p /batch_runs/new_shoji_full/plasma_params_0.plasma -n 0
exit 0

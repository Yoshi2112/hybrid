#!/bin/bash
#PBS -l select=1:ncpus=16:mem=15GB
#PBS -l walltime=50:00:00 
#PBS -k oe
#PBS -m ae
#PBS -M joshua.s.williams@uon.edu.au
# Autocreated by Python
source /etc/profile.d/modules.sh
module load numba/0.49.1-python.3.6
cd $PBS_O_WORKDIR
python /home/c3134027/hybrid/simulation_codes/PREDCORR_1D_PARALLEL/main_1D.py -r /batch_runs/JUL17_CO_1PC_1pc/run_params.run -p /batch_runs/JUL17_CO_1PC_1pc/pp5.plasma -n 5
exit 0

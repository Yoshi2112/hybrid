#!/bin/bash
# Created 24/06/2020 by c3134027
#PBS -l select=1:ncpus=8:mem=1GB
#PBS -l walltime=12:00:00           
#PBS -k oe
#PBS -m ae
#PBS -M joshua.s.williams@uon.edu.au
source /etc/profile.d/modules.sh
module load numba/0.49.1-python.3.6
cd $PBS_O_WORKDIR
python /home/c3134027/hybrid/simulation_codes/PREDCORR_1D_PARALLEL/main_1D.py -r /batch_runs/winske_periodic_uniform/run_params_PU256.run  -p /batch_runs/winske_periodic_uniform/plasma_params_c256_h1024.plasma -n 1
exit 0

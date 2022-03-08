#!/bin/bash
# Created 08/03/2022 by c3134027
#PBS -l select=1:ncpus=36:mem=20GB
#PBS -l walltime=200:00:00           
#PBS -k oe
#PBS -m ae
#PBS -M joshua.s.williams@uon.edu.au
source /etc/profile.d/modules.sh
module load numba/0.49.1-python.3.6
cd $PBS_O_WORKDIR
python /home/c3134027/hybrid/simulation_codes/PREDCORR_1D_PARALLEL/main_1D.py -r /batch_runs/CH4_tests/run_params_parabolic.run -p /batch_runs/CH4_tests/pp_Fu.plasma -n 6
exit 0
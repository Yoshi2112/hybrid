#!/bin/bash
# Created by c3134027, last update: 24/06/20.
#PBS -l select=1:ncpus=1:mem=2GB
#PBS -l walltime=12:00:00           
#PBS -k oe
#PBS –m bae
#PBS –M john.smith@newcastle.edu.au
source /etc/profile.d/modules.sh
module load numba/0.49.1-python.3.6
cd $PBS_O_WORKDIR
python main_1D.py
exit 0
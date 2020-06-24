#!/bin/bash
#
#PBS -l select=1:ncpus=1:mem=12gb
#PBS -l walltime=0:20:00
#PBS -l software=matlab
#PBS -k oe

# example job script to pass arguments in to matlab

source /etc/profile.d/modules.sh
module load matlab/R2019b

cd $PBS_O_WORKDIR

matlab -singleCompThread -nosplash -nodisplay -r "processargs('$VAR1','$VAR2','$OUTPUT');exit;"



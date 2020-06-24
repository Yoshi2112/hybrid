#!/bin/bash
#
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=20:00:00
#PBS -l software=matlab
#PBS -k oe

source /etc/profile.d/modules.sh
module load matlab/R2019b


cd $PBS_O_WORKDIR

matlab -singleCompThread -nodisplay -r "example1;exit;"



#!/bin/bash
#
#PBS -l select=1:ncpus=1:ngpus=1:mem=8gb
#PBS -l walltime=20:00:00
#PBS -l software=matlab
#PBS -k oe
#PBS -q gpuq

source /etc/profile.d/modules.sh
module load cuda10.1/toolkit
module load matlab


cd $PBS_O_WORKDIR

matlab -singleCompThread -nodisplay -r "gpuDeviceCount,gpuDevice(1),exit;"



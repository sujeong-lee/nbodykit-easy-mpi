# script to run Gattaca

#!/bin/bash
#PBS -N true
#PBS -q mpi-sn
###PBS -q debug-jpl
#PBS -l select=1:ncpus=40:mpiprocs=40:mem=10gb
#PBS -l walltime=10:00:00
#PBS -k oed
#PBS -o jobout/JOB_OUT.txt
#PBS -e jobout/JOB_ERR.txt
#PBS -W group_list=fiat.lux
#PBS -v summary=true


## your environment setting 
source ~/.bashrc
conda activate YOUR_ENV 
module load intel/mpi
# go to working dir
cd ${WORKING_DIR}

# launch with mpi
mpiexec python corr-mpi.py config/auto.yaml 1> jobout/JOB_OUT_${PBS_JOBID}.txt 2> jobout/JOB_ERR_${PBS_JOBID}.txt
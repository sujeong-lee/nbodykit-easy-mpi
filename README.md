# nbodykit-easy-mpi
Simple script to launch nbodykit correlation function computation with mpi 


```
mkdir jobout/

# for auto computation
mpiexec python corr-mpi.py config/auto.yaml 1> jobout/JOB_OUT_${PBS_JOBID}.txt 2> jobout/JOB_ERR_${PBS_JOBID}.txt
# for cross computation
mpiexec python corr-mpi.py config/cross.yaml 1> jobout/JOB_OUT_${PBS_JOBID}.txt 2> jobout/JOB_ERR_${PBS_JOBID}.txt
# for modified LS estimator
mpiexec python corr-mpi.py config/modified_estimator.yaml 1> jobout/JOB_OUT_${PBS_JOBID}.txt 2> jobout/JOB_ERR_${PBS_JOBID}.txt
```

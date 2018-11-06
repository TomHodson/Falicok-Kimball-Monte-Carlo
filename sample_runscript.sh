#!/usr/bin/env bash
#PBS -N MCMC_simulation
#PBS -lselect=1:ncpus=1:mem=1gb
#PBS -lwalltime=24:00:00
#PBS -J 1-{N_jobs}
#PBS -m abe
#PBS -M tch14@imperial.ac.uk

echo ------------------------------------------------------
echo -n 'Job is running on node '; cat $PBS_NODEFILE
echo ------------------------------------------------------
echo PBS: qsub is running on $PBS_O_HOST
echo PBS: originating queue is $PBS_O_QUEUE
echo PBS: executing queue is $PBS_QUEUE
echo PBS: working directory is $PBS_O_WORKDIR
echo PBS: execution mode is $PBS_ENVIRONMENT
echo PBS: job identifier is $PBS_JOBID
echo PBS: job name is $PBS_JOBNAME
echo PBS: node file is $PBS_NODEFILE
echo PBS: current home directory is $PBS_O_HOME
echo PBS: PATH = $PBS_O_PATH
echo ------------------------------------------------------

module load intel-suite anaconda3/personal
. /home/tch14/anaconda3/etc/profile.d/conda.sh
conda activate idp

cd {working_dir}
run_mcmc --job-id $PBS_ARRAY_INDEX --temp-dir $TMPDIR --working-dir ./
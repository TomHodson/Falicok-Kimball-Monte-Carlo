import numpy as np
from pathlib import Path
import h5py
import click
import time
from montecarlo import cython_mcmc


def config_dimensions(config):
    names = config['loop_over']
    vals = np.array([config[v] for v in names])
    lens = np.array([len(config[v]) for v in names])
    dtypes = np.array([config[v].dtype for v in names])
    return names, vals, lens, dtypes

def total_jobs(config):
    names, vals, lens, dtypes = config_dimensions(config)
    return lens.prod()

def get_config(job_id, configs):
    '''
    Take a job number and a config with multiple values and index into the cartesian product of configurations
    use the value 'loop_over' to decide which values get looped over
    '''
    assert(job_id < total_jobs(configs))
    #get all the keys that aren't looped over
    single_config= {k:v for k,v in configs.items() if k not in set(configs['loop_over'])}

    #starting from the innermost loop work outwards
    indices = []
    for key in configs['loop_over'][::-1]:
        values = configs[key]
        job_id, i  = divmod(job_id, len(values))
        single_config[key] = values[i]
        indices.append(i)

    indices = tuple(indices[::-1])

    single_config['job_id'] = job_id
    single_config['indices'] = indices

    return single_config

def setup_mcmc(mcmc_routine, config, working_dir = Path('./'), overwrite = False):
    working_dir.mkdir(parents=True, exist_ok=overwrite)
    (working_dir / 'jobs').mkdir(parents=True, exist_ok=True)

    names, vals, lens, dtypes = config_dimensions(config)
    data_shape = tuple(np.append(lens, config['N_steps']))

    first_config = get_config(job_id=0, configs=config)
    names, results = mcmc_routine(**first_config, sample_output = True)

    script = f'''
#!/usr/bin/env bash
#PBS -N MCMC_simulation
#PBS -lselect=1:ncpus=1:mem=1gb
#PBS -lwalltime=24:00:00
#PBS -J 1-{total_jobs(config)}
#PBS -m abe
#PBS -M tch14@imperial.ac.uk
#PBS -j oe
#PBS -o {working_dir.resolve()}/logs/$PBS_JOBID

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

run_mcmc --job-id $PBS_ARRAY_INDEX --working-dir {working_dir.resolve()}
    '''

    with open(working_dir / 'runscript.sh', 'w') as f:
        f.write(script)
        print(script)

    result_filename = working_dir / "results.hdf5"
    with h5py.File(result_filename, "w") as result_file:
        result_file.attrs.update(config)
        for name, val in zip(names, results):
            result_file.create_dataset(name, shape = data_shape, dtype = val.dtype)
        print(list(result_file.keys()))

@click.command()
@click.option('--job-id', default=1, help='which job to run')
@click.option('--working-dir', default='./', help='where to look for the config files')
@click.option('--temp-dir', default='./', help='where to store temporary files')
def run_mcmc(job_id, 
             working_dir = Path('./'),
             temp_dir = Path('./'),
             overwrite = True, mcmc_routine = cython_mcmc):
    '''
    Does the work that a single thread is expected to do.
    '''
    print(f'job_id == {job_id}')
    
    working_dir = Path(working_dir)
    result_file = working_dir / "results.hdf5"
    if not result_file.exists():
        print('No result file found')
        return

    with h5py.File(result_file, "r") as f:
        config = dict(f.attrs)

    starttime = time.time()
    this_config = get_config(job_id-1, config)
    names, results = mcmc_routine(**this_config)
    runtime = time.time() - starttime

    (working_dir / 'jobs').mkdir(exist_ok = True)
    job_file = working_dir / 'jobs' / f"job_{job_id}.hdf5"
    if job_file.exists() and overwrite == False:
        return

    with h5py.File(job_file, "w") as f:
        f.attrs.update(this_config)
        f.attrs['runtime'] = runtime
            
        for name, result in zip(names, results):
            result_data = f.create_dataset(name, data=result)

def job_completion(working_dir):
    result_filename = working_dir / "results.hdf5"
    job_dir =  working_dir / "jobs"

    missing = []
    with h5py.File(result_filename, "r+") as result_file:
        config = dict(result_file.attrs)

        for job_id in range(1,total_jobs(config)+1):
            job_filename = job_dir / f"job_{job_id}.hdf5"
            if not job_filename.exists():
                missing.append(job_id)
                
        print(f'missing : {len(missing)} of {total_jobs(config)} total')
        print(missing)
            
def gather_mcmc(working_dir, overwrite = True):
    result_filename = working_dir / "results.hdf5"
    job_dir =  working_dir / "jobs"

    missing = 0
    with h5py.File(result_filename, "r+") as result_file:
        config = dict(result_file.attrs)

        for job_id in range(1,total_jobs(config)+1):
            job_filename = job_dir / f"job_{job_id}.hdf5"
            if not job_filename.exists():
                missing += 1
                continue
            
            with h5py.File(job_filename, 'r') as job_file:
                #loop over the datasets, ie energy, magnetisation etc
                for dataset_name, val in job_file.items():
                    dataset = result_file[dataset_name]

                    indices = tuple(job_file.attrs['indices'])

                    #label each axis of the dataset
                    for dim,name in zip(dataset.dims,config['loop_over']):
                        dim.label = name

                    dataset[indices] = val
        print(f'missing : {missing} of {total_jobs(config)} total')

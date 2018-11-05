import numpy as np
from pathlib import Path
import h5py
import click
import time
from montecarlo import cython_mcmc
import logging


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

def setup_mcmc(mcmc_routine, config, working_dir = Path('./')):
    
    working_dir.mkdir(parents=True, exist_ok=True)
    (working_dir / 'jobs').mkdir(parents=True, exist_ok=True)
    (working_dir / 'logs').mkdir(parents=True, exist_ok=True)
    
    result_filename = working_dir / "results.hdf5"
    N_jobs = total_jobs(config)
    
    if result_filename.exists():
        #update the results file with the remaining jobs to be done
        with h5py.File(result_filename, "r+") as result_file:
            result_file.attrs['jobs_to_run'] = job_completion(working_dir)
            N_jobs = len(result_file.attrs['jobs_to_run'])
            print('results keys:', list(result_file.keys()))
    else:
        #create the results file and metadata from scratch
        names, vals, lens, dtypes = config_dimensions(config)
        data_shape = tuple(np.append(lens, config['N_steps']))

        first_config = get_config(job_id=0, configs=config)
        names, results = mcmc_routine(**first_config, sample_output = True)

        with h5py.File(result_filename, "w") as result_file:
            result_file.attrs.update(config)
            for name, val in zip(names, results):
                result_file.create_dataset(name, shape = data_shape, dtype = val.dtype)
        print(list(result_file.keys()))

    #update or create the script because the number of jobs to do might have changed
    script = open('./sample_runscript.sh').read().format(working_dir=working_dir.resolve(), N_jobs=N_jobs)
    with open(working_dir / 'runscript.sh', 'w') as f:
        f.write(script)
        print(script)

    


@click.command()
@click.option('--job-id', default=1, help='which job to run')
@click.option('--working-dir', default='./', help='where to look for the config files')
@click.option('--temp-dir', default='./', help='where to store temporary files')
def run_mcmc_command(*args, **kwargs):
    return run_mcmc(*args, **kwargs)

def run_mcmc(job_id, 
             working_dir = Path('./'),
             temp_dir = Path('./'),
             overwrite = False, mcmc_routine = cython_mcmc):
    '''
    Does the work that a single thread is expected to do.
    '''
    job_id = job_id - 1 #PBS array jobs are 1 based.
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f'job_id: {job_id}')
    
    working_dir = Path(working_dir)
    result_file = working_dir / "results.hdf5"
    if not result_file.exists():
        logger.info(f'No result file found')
        return
    
    (working_dir / 'jobs').mkdir(exist_ok = True)
    job_file = working_dir / 'jobs' / f"job_{job_id}.hdf5"
    if job_file.exists() and overwrite == False:
        logger.info(f'Job File already exists, not overwriting it')
        return

    with h5py.File(result_file, "r") as f:
        config = dict(f.attrs)
        
    if 'jobs_to_run' in config:
        job_id = config['jobs_to_run'][job_id]

    starttime = time.time()
    this_config = get_config(job_id, config)
    
    logger.info(f'This jobs config is {this_config}')
    logger.info(f'Starting MCMC routine')
    names, results = mcmc_routine(**this_config)
    runtime = time.time() - starttime
    logger.info(f'MCMC routine finished after {runtime:.2f} seconds')

    logger.info(f'Copying the results into the h5py file')
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
                
        return np.array(missing)
            
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
    print(f'File size: {result_filename.stat().st_size / 10**9}')

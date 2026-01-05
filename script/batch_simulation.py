from time import perf_counter
import numpy as np
from learn2assemble import default_settings
from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
from learn2assemble.simulator_parallel import ipm_simulate_parallel, ipm_init_simulate_parallel, ipm_split_states, \
    ipm_terminate, ipm_search_parameters_parallel, ipm_precondition
from learn2assemble.simulator import ipm_init, ipm_update_device
from learn2assemble.render import *
import torch
import os
import learn2assemble
import torch.multiprocessing as mp
from os.path import isfile, join, isdir
from os import listdir
import wandb
import platform
from tqdm import tqdm
from simulation_preload import parallel_load_assembly, is_wsl

if is_wsl():
    curriculumn_folder = "/mnt/d/curriculum_Thingi10K_12/Thingi10K_12/"
    assembly_folder = "/mnt/d/assembly_Thingi10K_12/Thingi10K_12/"
else:
    curriculumn_folder = "/scratch/assembly/curriculum/"
    assembly_folder = "/scratch/assembly/Thingi10K_12/"
result_table = []

def test_instance(sol_file, ipm_settings_cpu):
    obj_id = sol_file.split('_')[2]
    sol_id = sol_file.split('_')[4].split('.')[0]

    learn2assemble.simulator.logger = {
        'timer': {},
        'log': {},
        'activate': False
    }

    # load curriculum
    filename = os.path.join(curriculumn_folder, f"Thingi10K_12_{obj_id}_sol_{sol_id}.pt")
    part_states = torch.load(filename)['input']
    n_state = part_states.shape[0]
    print("num of states:", n_state)

    # gpus
    gpus = np.arange(torch.cuda.device_count())
    devices = [f"cuda:{gpu_id}" for gpu_id in gpus]

    # batch_size
    n_batch = 512

    # precondition
    ipm_settings = ipm_update_device(ipm_settings_cpu, 'cuda')
    ipm_precondition(ipm_settings)
    # back to cpu
    ipm_settings = ipm_update_device(ipm_settings, 'cpu')

    # init simulation
    simulators = ipm_init_simulate_parallel(ipm_settings, devices, n_batch)
    sim_datas = ipm_split_states(ipm_settings, part_states, n_batch)
    _, _, avg_sim_time, avg_success_rate = ipm_simulate_parallel(sim_datas, simulators)
    ipm_terminate(simulators)

    print("acc", avg_success_rate)
    print("time", avg_sim_time)

    result_table.append({"name": sol_file,
                         "n_parts": ipm_settings['n_part'],
                         "n_states": n_state,
                         "time": avg_sim_time,
                         "acc": avg_success_rate}
                        )

    # log wandb
    wandb.log({"n_parts": ipm_settings['n_part'],
               "n_states": n_state,
               "time": avg_sim_time,
               "acc": avg_success_rate}
              )

    with open('result.json', 'w') as f:
        json.dump(result_table, f, indent=4)

    return True

if __name__ == "__main__":
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        exit(0)

    sol_files = [f for f in listdir(curriculumn_folder) if isfile(join(curriculumn_folder, f))]
    sol_files.sort()
    sol_files = sol_files[::-1]
    sol_files = sol_files[:64]

    dict_ipm_settings = parallel_load_assembly(sol_files, n_worker=32)

    # setup wandb
    wandb.login(key="1c4a274de42ea0326b6ac75651a33f2b7cb2d217", relogin=True, force=True)
    run = wandb.init(project="Simulation", name="batch")

    with tqdm(total=len(sol_files), position=0) as progress:
        for sol_file in sol_files:
            print(sol_file)
            test_instance(sol_file, dict_ipm_settings[sol_file])
            progress.update()

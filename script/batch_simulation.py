from time import perf_counter

import numpy as np

from learn2assemble import default_settings
from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
from learn2assemble.simulator import ipm_init, ipm_search_parameters, ipm_update_device, ipm_simulate_parallel, ipm_get_states
from learn2assemble.render import *
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch
import platform
import os
import learn2assemble
import torch.multiprocessing as mp
from os.path import isfile, join, isdir
from os import listdir
import wandb

#curriculumn_folder = "/scratch/assembly/curriculum/"
#assembly_folder = "/scratch/assembly/Thingi10K_12/"
curriculumn_folder = "/mnt/d/curriculum_Thingi10K_12/Thingi10K_12/"
assembly_folder = "/mnt/d/assembly_Thingi10K_12/Thingi10K_12/"

result_table = []

def test_instance(obj_id, sol_id, devices=None):
    learn2assemble.simulator.logger = {
        'timer': {},
        'log': {},
        'activate': False
    }
    # test
    default_settings['rbe']['mu'] = 0.2
    default_settings['rbe']['velocity_tol'] = 1E-2
    default_settings["assembly"]["contact_shrink_ratio"] = 0.1  # for robustnessly computing the contact surfaces

    # ipm
    default_settings['ipm'] = {
        "n_iter": 30,
        "n_pcg_eval_iter": 10,
        "n_linesearch": 32,
        "kkt_conv_eps": 1E-4,
        "x_bound_tol": 1E-5,
        "float_type": torch.float32,
    }

    # load geometry
    foldername = os.path.join(assembly_folder, f"Thingi10K_12_{obj_id}/sol_{sol_id}")
    parts = load_assembly_from_files(foldername)

    # decided batch size
    if devices is None:
        gpus = np.arange(torch.cuda.device_count())
        devices = [f"cuda:{gpu_id}" for gpu_id in gpus]
    n_gpu = len(devices)

    # for h800
    n_batch = 2048
    n_batch *= n_gpu

    # compute contacts
    contacts = compute_assembly_contacts(parts, default_settings)
    ipm_settings = ipm_init(parts, contacts, default_settings)

    # load curriculum
    filename = os.path.join(curriculumn_folder, f"Thingi10K_12_{obj_id}_sol_{sol_id}.pt")
    part_states = torch.load(filename)['input']
    n_state = part_states.shape[0]
    print("num of states:", n_state)

    # search best parameters
    if not ipm_search_parameters(ipm_settings, part_states, 512, 0.9):
        return False
    # update settings
    ipm_settings_cpu = ipm_update_device(ipm_settings, 'cpu')

    dataloader = DataLoader(
        TensorDataset(part_states),
        batch_size=n_batch,  # How many samples per batch
        shuffle=False,  # Shuffle data every epoch
        num_workers=2  # Use 2 subprocesses for loading (adjust as needed)
    )

    torch.cuda.synchronize()
    start_timer = perf_counter()

    tot_success = 0
    with tqdm(total=len(dataloader)) as progress:
        for part_states in dataloader:
            # padding
            part_states = part_states[0]
            test_states, n_test_sub = ipm_get_states(part_states, ipm_settings['boundary_part_ids'], n_sample=n_batch)
            # simulation
            _, stable_fp32 = ipm_simulate_parallel(test_states, ipm_settings_cpu, devices)
            stable_fp32 = stable_fp32[: n_test_sub]
            # evaluation
            tot_success += torch.sum(stable_fp32).item()
            progress.set_postfix_str(torch.sum(stable_fp32) / stable_fp32.shape[0])
            progress.update()

    torch.cuda.synchronize()
    print("time:\t", perf_counter() - start_timer)
    print("success rate:\t", tot_success / n_state)
    print("\n")

    result_table.append({"name": obj_id,
                         "sol_id": sol_id,
                         "n_parts": len(parts),
                         "n_states": n_state,
                         "time": perf_counter() - start_timer,
                         "acc": tot_success / n_state}
                        )

    # log wandb
    wandb.log({"n_parts": len(parts),
               "n_states": n_state,
               "time": perf_counter() - start_timer,
               "acc": tot_success / n_state}
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
    gpus = np.arange(torch.cuda.device_count())
    devices = [f"cuda:{gpu_id}" for gpu_id in gpus]
    print("devices:", devices)

    # setup wandb
    wandb.login(key="1c4a274de42ea0326b6ac75651a33f2b7cb2d217", relogin=True, force=True)
    run = wandb.init(project="Simulation", name="batch")

    sol_files = [f for f in listdir(curriculumn_folder) if isfile(join(curriculumn_folder, f))]
    sol_files.sort()
    sol_files = sol_files[::-1]
    for sol_file in sol_files:
        obj_id = sol_file.split('_')[2]
        sol_id = sol_file.split('_')[4].split('.')[0]
        print(obj_id, sol_id)
        test_instance(obj_id, sol_id, devices=devices)

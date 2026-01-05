from time import perf_counter
from pympler import asizeof
from learn2assemble import default_settings
from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
from learn2assemble.simulator import ipm_init, ipm_search_parameters, ipm_simulate, ipm_get_states, print_logger, \
    ipm_update_device, ipm_precondition
from learn2assemble.render import *
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch
import platform
import os
import learn2assemble
from os.path import isfile, join, isdir
from os import listdir
import sys
import copy
import torch.multiprocessing as mp

def is_wsl():
    # 'uname -r' equivalent
    release = platform.release().lower()
    return 'microsoft' in release or 'wsl' in release

if is_wsl():
    curriculumn_folder = "/mnt/d/curriculum_Thingi10K_12/Thingi10K_12/"
    assembly_folder = "/mnt/d/assembly_Thingi10K_12/Thingi10K_12/"
if platform.system() == 'Windows':
    curriculumn_folder = "D:/curriculum_Thingi10K_12/Thingi10K_12"
    assembly_folder = "D:/assembly_Thingi10K_12/Thingi10K_12"
elif is_wsl():
    curriculumn_folder = "/mnt/d/curriculum_Thingi10K_12/Thingi10K_12/"
    assembly_folder = "/mnt/d/assembly_Thingi10K_12/Thingi10K_12/"
else:
    curriculumn_folder = "/scratch/assembly/curriculum/"
    assembly_folder = "/scratch/assembly/Thingi10K_12/"

result_table = []

def compute_memory(ipm_settings):
    total_memory = 0
    for name, val in ipm_settings.items():
        if torch.is_tensor(val):
            total_memory += val.nelement() * val.element_size()
    return total_memory / 1024.0 / 1024.0

def load_assembly(in_, out_, return_dict):
    ipm_settings_default = copy.deepcopy(default_settings)
    # default settings
    ipm_settings_default['rbe']['mu'] = 0.2
    ipm_settings_default['rbe']['velocity_tol'] = 1E-2
    ipm_settings_default["assembly"]["contact_shrink_ratio"] = 0.1  # for robustnessly computing the contact surfaces
    ipm_settings_default['ipm'] = {
        "n_iter": 30,
        "n_pcg_eval_iter": 10,
        "n_pcg_iter": 300,
        "n_linesearch": 32,
        "kkt_conv_eps": 1E-4,
        "x_bound_tol": 1E-5,
        "float_type": torch.float32,
        "device": "cpu"
    }

    while True:
        sol_file = in_.get()

        obj_id = sol_file.split('_')[2]
        sol_id = sol_file.split('_')[4].split('.')[0]

        if sol_file is None:
            break

        # load geometry
        foldername = os.path.join(assembly_folder, f"Thingi10K_12_{obj_id}/sol_{sol_id}")
        parts = load_assembly_from_files(foldername)

        # compute contacts
        contacts = compute_assembly_contacts(parts, ipm_settings_default)

        # init ipm
        ipm_settings = ipm_init(parts, contacts, ipm_settings_default)

        # send back
        return_dict[sol_file] = ipm_settings
        out_.put((sol_file, compute_memory(ipm_settings)))

def test_instance(sol_file, ipm_settings_cpu):
    torch.cuda.synchronize()
    start_timer = perf_counter()

    obj_id = sol_file.split('_')[2]
    sol_id = sol_file.split('_')[4].split('.')[0]

    learn2assemble.simulator.logger = {
        'timer': {},
        'log': {},
        'activate': False
    }

    ipm_settings = ipm_update_device(ipm_settings_cpu, 'cuda')
    ipm_precondition(ipm_settings)

    # decided batch size
    n_batch = 1024
    if ipm_settings_cpu['n_part'] > 80:
        n_batch = 512

    # load curriculum
    filename = os.path.join(curriculumn_folder, f"Thingi10K_12_{obj_id}_sol_{sol_id}.pt")
    part_states = torch.load(filename)['input']
    n_state = part_states.shape[0]
    print("num of states:", n_state)

    dataloader = DataLoader(
        TensorDataset(part_states),
        batch_size=n_batch,  # How many samples per batch
        shuffle=False,  # Shuffle data every epoch
        num_workers=2  # Use 2 subprocesses for loading (adjust as needed)
    )


    tot_success = 0
    with tqdm(total=len(dataloader), position=1) as progress:
        for part_states in dataloader:
            # padding
            part_states = part_states[0]
            test_states, n_test_sub = ipm_get_states(part_states, ipm_settings['boundary_part_ids'], n_sample=n_batch)
            # simulation
            _, stable_fp32 = ipm_simulate(test_states, ipm_settings)
            stable_fp32 = stable_fp32[: n_test_sub]
            # evaluation
            tot_success += torch.sum(stable_fp32).item()
            progress.set_postfix_str(torch.sum(stable_fp32) / stable_fp32.shape[0])
            progress.update()

    torch.cuda.synchronize()
    print("time:\t", (perf_counter() - start_timer) / n_batch)
    print("success rate:\t", tot_success / n_state)
    print("\n")

    result_table.append({"name": obj_id,
                         "sol_id": sol_id,
                         "n_parts": ipm_settings['n_part'],
                         "n_states": n_state,
                         "time": (perf_counter() - start_timer) / n_batch,
                         "acc": tot_success / n_state}
                        )

    with open('result.json', 'w') as f:
        json.dump(result_table, f, indent=4)

    return True

def parallel_load_assembly(sol_files, n_worker = 64):
    manager = mp.Manager()
    dict_ipm_settings = manager.dict()
    in_ = manager.Queue()
    out_ = manager.Queue()

    jobs = []
    for worker in range(n_worker):
        p = mp.Process(target=load_assembly,
                       args=(in_, out_, dict_ipm_settings))
        jobs.append(p)
        p.start()

    for sol_file in sol_files:
        in_.put(sol_file)

    # preloading
    num_done = 0
    with tqdm(total=len(sol_files)) as progress:
        while num_done < len(sol_files):
            sol_file, memory = out_.get()
            num_done += 1
            progress.set_postfix_str(f"{memory:.2f} MB")
            progress.update()

    # exit
    for sol_file in sol_files:
        in_.put(None)

    for p in jobs:
        p.join()

    return dict(dict_ipm_settings)


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

    #sol_files = sol_files[:10]

    # skip
    # for id in range(len(sol_files)):
    #     sol_file = sol_files[id]
    #     obj_id = sol_file.split('_')[2]
    #     sol_id = sol_file.split('_')[4].split('.')[0]
    #     if obj_id == "2426":
    #         sol_files = sol_files[id:-1]
    #         break


    dict_ipm_settings = parallel_load_assembly(sol_files, n_worker=64)

    with tqdm(total=len(sol_files), position=0) as progress:
        for sol_file in sol_files:
            test_instance(sol_file, dict_ipm_settings[sol_file])
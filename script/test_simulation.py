from torch.utils.data import Dataset

from learn2assemble import ASSEMBLY_RESOURCE_DIR, default_settings, RESOURCE_DIR
from learn2assemble.render import *
from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
from learn2assemble.simulator import init_ipm, init_rbe, print_logger, simulate
import torch
import os
from learn2assemble.render import *
import polyscope as ps
from torch.utils.data import DataLoader, TensorDataset

curriculumn_folder = "/mnt/d/curriculum_Thingi10K_12/Thingi10K_12/"
assembly_folder = "/mnt/d/assembly_Thingi10K_12/Thingi10K_12/"
from os import listdir
from os.path import isfile, join
import learn2assemble

def test_instance(obj_id, sol_id, ipm=True):
    learn2assemble.simulator.logger = {
        'timer': {},
        'log': {},
        'activate': True
    }
    # test
    default_settings['rbe']['density'] = 1
    default_settings['rbe']['mu'] = 0.2
    default_settings['rbe']['Ccp'] = 5
    default_settings['rbe']['velocity_tol'] = 1E-2
    default_settings["assembly"]["contact_shrink_ratio"] = 0.1  # for robustnessly computing the contact surfaces

    # ipm
    if not ipm:
        default_settings['gurobi'] = {}

    default_settings['ipm'] = {
        "n_iter": 30,
        "n_pcg_iter_1": 200,
        "n_pcg_iter_2": 50,
        "n_pcg_eval_iter": 10,
        "n_linesearch": 32,
        "kkt_conv_eps": 1E-5,
        "x_bound_tol": 1E-6,
        "float_type": torch.float32,
        "use_Q_fast": True,
    }

    n_batch = 2048

    # load geometry
    foldername = os.path.join(assembly_folder, f"Thingi10K_12_{obj_id}/sol_{sol_id}")
    parts = load_assembly_from_files(foldername)
    contacts = compute_assembly_contacts(parts, default_settings)
    init_rbe(parts, contacts, default_settings)
    init_ipm(parts, contacts, default_settings)
    print("n_parts:\t", len(parts))

    # load curriculum
    filename = os.path.join(curriculumn_folder, f"Thingi10K_12_{obj_id}_sol_{sol_id}.pt")
    part_states = torch.load(filename)['input']

    # try best parameters
    n_pcg_iter_1 = [50, 100, 150, 200, 250, 300, 350, 400]
    n_sample = 32
    for attempt_iter in n_pcg_iter_1:
        default_settings['ipm']['n_pcg_iter_1'] = attempt_iter
        v_fp32, stable_fp32 = simulate(parts, contacts, part_states[-n_sample:, :], default_settings)
        print("attempt success rate", np.sum(stable_fp32) / n_sample)
        if np.sum(stable_fp32) > n_sample * 0.9:
            print("use n_pcg_iter_1 = ", attempt_iter)
            break

    inds = torch.sum(part_states, dim=1).cpu().numpy()
    inds = torch.tensor(np.argsort(inds).tolist())
    part_states = part_states[inds, :]

    dataloader = DataLoader(
        TensorDataset(part_states),
        batch_size=n_batch,  # How many samples per batch
        shuffle=False,  # Shuffle data every epoch
        num_workers=2  # Use 2 subprocesses for loading (adjust as needed)
    )

    tot_success = 0
    for part_states in dataloader:
        part_states = part_states[0]
        v_fp32, stable_fp32 = simulate(parts, contacts, part_states, default_settings)
        tot_success += np.sum(stable_fp32)
    print("success rate:\t", tot_success / inds.shape[0])
    print_logger(['ipm', 'gurobi'])
    print("\n")

sol_files = [f for f in listdir(curriculumn_folder) if isfile(join(curriculumn_folder, f))]
for sol_file in sol_files:
    obj_id = sol_file.split('_')[2]
    sol_id = sol_file.split('_')[4].split('.')[0]
    print(obj_id, sol_id)
    test_instance(obj_id, sol_id, True)

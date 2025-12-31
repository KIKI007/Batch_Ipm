from learn2assemble import default_settings
from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
from learn2assemble.simulator import init_ipm, print_logger, simulate, ipm_search_best_parameters
from learn2assemble.render import *
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch
import platform
import os
from os import listdir
from os.path import isfile, join
import learn2assemble

if platform.system() == 'Windows':
    curriculumn_folder = "D:/curriculum_Thingi10K_12/Thingi10K_12"
    assembly_folder = "D:/assembly_Thingi10K_12/Thingi10K_12"
else:
    curriculumn_folder = "/mnt/d/curriculum_Thingi10K_12/Thingi10K_12/"
    assembly_folder = "/mnt/d/assembly_Thingi10K_12/Thingi10K_12/"

result_table = []

def test_instance(obj_id, sol_id, ipm=True):
    learn2assemble.simulator.logger = {
        'timer': {},
        'log': {},
        'activate': True
    }
    # test
    default_settings['rbe']['mu'] = 0.2
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

    # load geometry
    foldername = os.path.join(assembly_folder, f"Thingi10K_12_{obj_id}/sol_{sol_id}")
    parts = load_assembly_from_files(foldername)

    # decided batch size
    n_batch = 1024
    if len(parts) > 80:
        n_batch = 512

    # compute contacts
    contacts = compute_assembly_contacts(parts, default_settings)
    init_ipm(parts, contacts, default_settings)

    # load curriculum
    filename = os.path.join(curriculumn_folder, f"Thingi10K_12_{obj_id}_sol_{sol_id}.pt")
    part_states = torch.load(filename)['input']
    inds = torch.sum(part_states, dim=1).cpu().numpy()
    inds = torch.tensor(np.argsort(inds).tolist())
    part_states = part_states[inds, :]

    # search best parameters
    ipm_search_best_parameters(part_states[-32:], 0.8, default_settings)

    dataloader = DataLoader(
        TensorDataset(part_states),
        batch_size=n_batch,  # How many samples per batch
        shuffle=False,  # Shuffle data every epoch
        num_workers=2  # Use 2 subprocesses for loading (adjust as needed)
    )

    tot_success = 0
    with tqdm(total=len(dataloader)) as progress:
        for part_states in dataloader:
            part_states = part_states[0]
            v_fp32, stable_fp32 = learn2assemble.simulator.simulate(parts, contacts, part_states, default_settings)
            tot_success += np.sum(stable_fp32)
            progress.set_postfix_str(np.sum(stable_fp32) / stable_fp32.shape[0])
            progress.update()
    print("success rate:\t", tot_success / inds.shape[0])
    print_logger(inds.shape[0], ['ipm', 'gurobi'])
    print("\n")

    result_table.append({"name": obj_id,
                         "sol_id": sol_id,
                         "n_parts": len(parts),
                         "n_states": inds.shape[0],
                         "time": learn2assemble.simulator.logger['log']['ipm'],
                         "acc": tot_success / inds.shape[0]}
                        )

    with open('result.json', 'w') as f:
        json.dump(result_table, f, indent=4)

# sol_files = [f for f in listdir(curriculumn_folder) if isfile(join(curriculumn_folder, f))]
# sol_files.sort()
# for sol_file in sol_files:
#     obj_id = sol_file.split('_')[2]
#     sol_id = sol_file.split('_')[4].split('.')[0]
#     print(obj_id, sol_id)
#     test_instance(obj_id, sol_id, True)
test_instance(1026, 0)
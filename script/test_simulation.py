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

def test_instance(obj_id, sol_id):

    # test
    default_settings['rbe']['density'] = 10
    default_settings['rbe']['mu'] = 0.2
    default_settings['rbe']['Ccp'] = 50
    default_settings["assembly"]["contact_shrink_ratio"] = 0.1  # for robustnessly computing the contact surfaces

    # ipm
    default_settings['ipm'] = {
        "n_iter": 30,
        "n_pcg_iter_1": 100,
        "n_pcg_iter_2": 50,
        "n_pcg_eval_iter": 10,
        "n_linesearch": 32,
        "kkt_conv_eps": 1E-5,
        "x_bound_tol": 1E-6,
        "float_type": torch.float32,
        "use_Q_fast": True,
    }

    n_batch = 512
    torch.manual_seed(0)

    # load geometry
    foldername = os.path.join("/mnt/d/assembly_Thingi10K_12/Thingi10K_12", f"Thingi10K_12_{obj_id}/sol_{sol_id}")
    parts = load_assembly_from_files(foldername)
    contacts = compute_assembly_contacts(parts, default_settings)
    init_rbe(parts, contacts, default_settings)
    init_ipm(parts, contacts, default_settings)

    # load curriculum
    filename = os.path.join("/mnt/d/curriculum_Thingi10K_12/Thingi10K_12", f"Thingi10K_12_{obj_id}_sol_{sol_id}.pt")
    part_states = torch.load(filename)['input']
    init_polyscope()
    inds = torch.sum(part_states, dim=1).cpu().numpy()
    inds = torch.tensor(np.argsort(inds).tolist())
    part_states = part_states[inds, :]

    dataloader = DataLoader(
        TensorDataset(part_states),
        batch_size=512,  # How many samples per batch
        shuffle=False,  # Shuffle data every epoch
        num_workers=2  # Use 2 subprocesses for loading (adjust as needed)
    )

    for part_states in dataloader:
        part_states = part_states[0]
        v_fp32, stable_fp32 = simulate(parts, contacts, part_states[0], default_settings)
        print_logger(part_states.shape[0])
        print(np.sum(stable_fp32) / n_batch)

test_instance(1, 0)
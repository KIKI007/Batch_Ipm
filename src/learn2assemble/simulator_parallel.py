import copy
from time import perf_counter
import gurobipy as gp
import torch
from gurobipy import GRB
from sympy.physics.units import velocity

from learn2assemble.rbe import *
from types import SimpleNamespace
import platform
from learn2assemble.rbe import num_vars
import torch.multiprocessing as mp
from learn2assemble.simulator import *


def ipm_simulate_parallel_proc(device_str,
                               ipm_settings_cpu,
                               n_batch,
                               in_queue: mp.Queue,
                               out_queue: mp.Queue):
    # warm up
    torch.set_float32_matmul_precision('high')
    device = torch.device(device_str)
    ipm_settings = ipm_update_device(ipm_settings_cpu, device)
    ipm_compile_functions(ipm_settings, False)
    warm_states = ipm_empty_states(ipm_settings['n_part'], ipm_settings['boundary_part_ids'], n_batch)
    ipm_warmup(warm_states, ipm_settings)
    out_queue.put(f"{device_str}: Warmup Done")

    # compute
    while True:
        data = in_queue.get()
        if data is None:
            break
        job_id, batch_part_states, n_sub_states = data
        velocity, flag = ipm_simulate(batch_part_states, ipm_settings)
        velocity = velocity[:n_sub_states, :]
        flag = flag[:n_sub_states]
        out_queue.put((job_id, velocity.cpu(), flag.cpu()))
    return

def ipm_simulate_parallel(batch_part_states: torch.tensor,
                          ipm_settings_cpu,
                          devices: list[str],
                          n_batch):
    batch_part_states = batch_part_states.to(device='cpu')

    n_parallel = len(devices)
    manager = mp.Manager()

    in_queue = manager.Queue()
    out_queue = manager.Queue()

    # init n_parallel process
    jobs = []
    for id in range(n_parallel):
        device = devices[id]
        p = mp.Process(target=ipm_simulate_parallel_proc,
                       args=(device,
                             ipm_settings_cpu,
                             n_batch,
                             in_queue,
                             out_queue))
        jobs.append(p)
        p.start()

    # wait until warmup
    num_done = 0
    while num_done < n_parallel:
        done = out_queue.get()
        print(done)
        num_done += 1

    # start sending data
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timer = perf_counter()

    n_state = batch_part_states.shape[0]
    n_step = batch_part_states.shape[0] // n_batch
    if n_state % n_batch != 0:
        n_step = n_step + 1

    for id in range(n_step):
        if id != n_step - 1:
            inds = torch.arange(id * n_batch,
                                n_batch * (id + 1),
                                device='cpu',
                                dtype=torch.long)
        else:
            # last take all
            inds = torch.arange(id * n_batch,
                                batch_part_states.shape[0],
                                device='cpu',
                                dtype=torch.long)
        part_states = batch_part_states[inds, :]
        part_states, n_sub_states = ipm_get_states(part_states, ipm_settings_cpu['boundary_part_ids'], n_batch)
        in_queue.put((id, part_states, n_sub_states))

    # for stop the solver
    for id in range(n_parallel):
        in_queue.put(None)

    # terminate all process
    for proc in jobs:
        proc.join()

    # collect data
    return_dict = {}
    for id in range(n_parallel):
        job_id, velocity, stable_flag = out_queue.get()
        return_dict[job_id] = (velocity, stable_flag)

    velocity = []
    stable_flag = []
    for id in range(n_parallel):
        velocity.append(return_dict[id][0])
        stable_flag.append(return_dict[id][1])

    velocity = torch.hstack(velocity)
    stable_flag = torch.hstack(stable_flag)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sim_time = perf_counter() - timer

    print("time ", sim_time / stable_flag.shape[0])
    return velocity, stable_flag

if __name__ == '__main__':
    from learn2assemble import ASSEMBLY_RESOURCE_DIR, default_settings, RESOURCE_DIR
    from learn2assemble.render import *
    from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
    import os

    default_settings['rbe']['mu'] = 0.2
    default_settings["assembly"]["contact_shrink_ratio"] = 0.1  # for robustnessly computing the contact surfaces

    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        exit(0)

    n_batch = 2048
    torch.manual_seed(0)
    name = "tetris-999"
    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + f"/{name}")
    boundary = [len(parts) - 1]
    default_settings['env']['boundary_part_ids'] = boundary

    filename = os.path.join(RESOURCE_DIR, f"curriculum/{name}.pt")
    part_states = torch.load(filename)['input']
    # part_states[:, 10] = 0
    # part_states[:, 31] = 0

    # sample
    part_states = ipm_sort_states(part_states, False)
    part_states, _ = ipm_get_states(part_states, boundary, n_batch)

    # default_settings['gurobi'] = {}
    default_settings['ipm'] = {
        "n_iter": 25,
        "n_pcg_iter": 200,
        "n_pcg_eval_iter": 10,
        "x_bound_tol": 1E-5,
        "kkt_conv_eps": 1E-4,
        "float_type": torch.float32,
    }

    contacts = compute_assembly_contacts(parts, default_settings)

    ipm_settings = ipm_init(parts, contacts, default_settings)
    ipm_settings_cpu = ipm_update_device(ipm_settings, 'cpu')

    gpus = np.arange(torch.cuda.device_count())
    devices = [f"cuda:{gpu_id}" for gpu_id in gpus]

    v_fp32, stable_fp32 = ipm_simulate_parallel(part_states, ipm_settings_cpu, devices, 2048)
    print(torch.sum(stable_fp32).item() / stable_fp32.shape[0])

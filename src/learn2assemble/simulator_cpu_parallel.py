import numpy as np
from learn2assemble.simulator import *
from tqdm import tqdm
from simulator import *
import multiprocessing as mp

def gurobi_simulate_parallel_proc(job_id,
                                  settings_cpu,
                                  test_states,
                                  return_dict):
    settings = copy.deepcopy(settings_cpu)
    init_gurobi(settings)
    vel, flags = gurobi_simulate(test_states, settings)
    return_dict[job_id] = (vel.cpu(), flags.cpu())
    return

def gurobi_simulate_parallel(batch_part_states, settings: dict):
    n_parallel = settings['gurobi'].get('nsim', 32)

    manager = mp.Manager()
    return_dict = manager.dict()

    # init n_parallel process
    jobs = []
    n_batch = batch_part_states.shape[0] // n_parallel

    for id in range(n_parallel):
        if id == n_parallel - 1:
            test_states = batch_part_states[id * n_batch : , :]
        else:
            test_states = batch_part_states[id * n_batch : (id + 1) * n_batch, :]

        p = mp.Process(target=gurobi_simulate_parallel_proc,
                       args=(id, settings, test_states, return_dict))
        jobs.append(p)
        p.start()

    # read result
    timer = perf_counter()

    # terminate all process
    for proc in jobs:
        proc.join()

    velocity = []
    stable_flag = []
    for id in range(n_parallel):
        velocity.append(return_dict[id][0])
        stable_flag.append(return_dict[id][1])
    velocity = torch.hstack(velocity)
    stable_flag = torch.hstack(stable_flag)

    print("avg time", (perf_counter() - timer) / stable_flag.shape[0])
    return velocity, stable_flag

if __name__ == '__main__':
    from learn2assemble import ASSEMBLY_RESOURCE_DIR, default_settings, RESOURCE_DIR
    from learn2assemble.render import *
    from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
    import os
    num_cores = os.cpu_count() * 2
    print("num_threads", num_cores)

    default_settings['rbe']['mu'] = 0.2
    default_settings["assembly"]["contact_shrink_ratio"] = 0.1  # for robustnessly computing the contact surfaces

    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

    n_batch = 2048
    torch.manual_seed(0)
    name = "tetris-999"
    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + f"/{name}")
    boundary = [0]
    default_settings['gurobi'] = {"nsim": num_cores}
    default_settings['env']['boundary_part_ids'] = boundary

    filename = os.path.join(RESOURCE_DIR, f"curriculum/{name}.pt")
    part_states = torch.load(filename)['input']

    # sample
    part_states = ipm_sort_states(part_states, False)
    part_states, _ = ipm_get_states(part_states, boundary, n_batch)

    # check
    contacts = compute_assembly_contacts(parts, default_settings)
    init_rbe(parts, contacts, default_settings)

    v_fp32, stable_fp32 = gurobi_simulate_parallel(part_states, default_settings)

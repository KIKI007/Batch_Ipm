import numpy as np
from learn2assemble.simulator import *
from tqdm import tqdm

def ipm_warmup(batch_part_states: torch.tensor, settings: dict):
    active = logger['activate']
    logger['activate'] = False
    n_iter = settings['n_iter']
    n_pcg_iter = settings['n_pcg_iter']
    settings["n_iter"] = 1
    settings["n_pcg_iter"] = 1
    ipm_simulate(batch_part_states, settings)
    settings["n_pcg_iter"] = n_pcg_iter
    settings["n_iter"] = n_iter
    logger['activate'] = active

def ipm_simulate_parallel_proc(device_str,
                               ipm_settings_cpu,
                               n_batch,
                               in_queue: mp.Queue,
                               out_queue: mp.Queue):
    # warm up
    torch.set_float32_matmul_precision('high')
    device = torch.device(device_str)
    ipm_settings = ipm_update_device(ipm_settings_cpu, device)
    ipm_precondition(ipm_settings)
    warm_states = ipm_empty_states(ipm_settings['n_part'], ipm_settings['boundary_part_ids'], n_batch)
    ipm_warmup(warm_states, ipm_settings)
    out_queue.put(f"{device_str}: Warmup Done")

    # compute
    while True:
        data = in_queue.get()
        if data is None:
            break
        job_id, batch_part_states, n_sub_states, n_pcg_iter = data
        ipm_settings['n_pcg_iter'] = n_pcg_iter
        velocity, flag = ipm_simulate(batch_part_states, ipm_settings)
        velocity = velocity[:n_sub_states, :]
        flag = flag[:n_sub_states]
        out_queue.put((job_id, velocity.cpu(), flag.cpu()))
    return

def ipm_init_simulate_parallel(ipm_settings_cpu,
                               devices: list[str],
                               n_batch):
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
        num_done += 1

    simulators = (jobs, in_queue, out_queue)
    return simulators

def ipm_split_states(ipm_settings_cpu,
                     batch_part_states,
                     n_batch):
    n_state = batch_part_states.shape[0]
    n_step = batch_part_states.shape[0] // n_batch
    if n_state % n_batch != 0:
        n_step = n_step + 1

    sim_datas = []
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
        sim_datas.append((id, part_states, n_sub_states, ipm_settings_cpu['n_pcg_iter']))
    return sim_datas


def ipm_terminate(simulators):
    jobs, in_queue, out_queue = simulators
    n_parallel = len(jobs)
    # for stop the solver
    for id in range(n_parallel):
        in_queue.put(None)

    # terminate all process
    for proc in jobs:
        proc.join()

def ipm_simulate_parallel(sim_datas, simulators):
    jobs, in_queue, out_queue = simulators

    # start sending data
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timer = perf_counter()

    # load data
    n_step = len(sim_datas)
    for data in sim_datas:
        id, part_states, n_sub_states, n_pcg = data
        in_queue.put((id, part_states, n_sub_states, n_pcg))

    # read the output
    return_dict = {}
    num_done = 0
    with tqdm(total=n_step, position=1) as progress:
        while num_done < n_step:
            job_id, velocity, stable_flag = out_queue.get()
            return_dict[job_id] = (velocity, stable_flag)
            curr_acc = torch.sum(stable_flag).item() / stable_flag.shape[0]
            progress.set_postfix_str(f"{curr_acc: .3f}")
            progress.update()
            num_done = num_done + 1
    print("\n")

    velocity = []
    stable_flag = []
    for id in range(n_step):
        velocity.append(return_dict[id][0])
        stable_flag.append(return_dict[id][1])
    velocity = torch.vstack(velocity)
    stable_flag = torch.hstack(stable_flag)
    n_state = stable_flag.shape[0]

    # print time and acc
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    avg_sim_time = (perf_counter() - timer) / n_state
    avg_success_rate = torch.sum(stable_flag).item() / n_state
    return velocity, stable_flag, avg_sim_time, avg_success_rate

def ipm_search_parameters_parallel(ipm_settings: dict,
                                   part_states,
                                   simulators,
                                   nsample=32,
                                   acc_tol=0.9):
    # load data
    new_states = ipm_sort_states(part_states, ascend=False)
    test_states, n_test_sub = ipm_get_states(new_states, ipm_settings['boundary_part_ids'], n_sample=nsample)
    n_pcg_it = ipm_settings["n_pcg_iter"]
    best_acc = 0.0
    sim_datas = []
    scale_list = [1, 1.5, 2, 2.5, 3, 3.5, 4]
    #scale_list = [1, 2]
    for id, scale in enumerate(scale_list):
        n_pcg_iter = int(n_pcg_it * scale)
        sim_datas.append((id, test_states, n_test_sub, n_pcg_iter))

    # simulate
    v, flag, x, y = ipm_simulate_parallel(sim_datas, simulators)
    flag = flag.reshape(-1, n_test_sub)
    acc = torch.sum(flag, dim = 1) / n_test_sub
    print("acc:", acc)

    if (acc > acc_tol).any():
        indices = torch.arange(len(sim_datas))
        indices = indices[acc > acc_tol]
        index = torch.min(indices)
        best_acc = acc[index]
        n_pcg_iter = sim_datas[index][3]
        ipm_settings["n_pcg_iter"] = n_pcg_iter
        print("num pcg iter = ", ipm_settings["n_pcg_iter"], f" with a {best_acc: .2f} success rate")
        return True
    else:
        print(f"Failed to find pcg iter with a maximum {best_acc: .2f} success rate")
        return False


def gurobi_simulate_parallel_proc(parts, contacts,
                               settings_cpu,
                               in_queue: mp.Queue,
                               out_queue: mp.Queue):

    settings = copy.deepcopy(settings_cpu)
    init_gurobi(settings)

    # compute
    while True:
        data = in_queue.get()
        if data is None:
            break
        job_id, part_states = data
        part_states = part_states.reshape(1, -1)
        velocity, flags = gurobi_simulate(part_states, settings)
        out_queue.put((job_id, velocity.cpu(), flags.cpu()))
    return


def gurobi_simulate_parallel(batch_part_states: list[dict], settings: dict):
    n_parallel = settings['gurobi'].get('nsim', 32)

    manager = mp.Manager()
    in_queue = manager.Queue()
    out_queue = manager.Queue()

    # init n_parallel process
    jobs = []
    for id in range(n_parallel):
        p = mp.Process(target=gurobi_simulate_parallel_proc,
                       args=(settings,
                             in_queue,
                             out_queue))
        jobs.append(p)
        p.start()
        print(f"start process {id}")

    # start simulation
    n_step = batch_part_states.shape[0]
    for id in range(n_step):
        in_queue.put((id, batch_part_states[id, :]))

    return_dict = {}
    num_done = 0
    with tqdm(total=n_step, position=1) as progress:
        while num_done < n_step:
            job_id, velocity, stable_flag = out_queue.get()
            return_dict[job_id] = (velocity, stable_flag)
            curr_acc = torch.sum(stable_flag).item() / stable_flag.shape[0]
            progress.set_postfix_str(f"{curr_acc: .3f}")
            progress.update()
            num_done = num_done + 1

    velocity = []
    stable_flag = []
    for id in range(n_step):
        velocity.append(return_dict[id][0])
        stable_flag.append(return_dict[id][1])
    velocity = torch.vstack(velocity)
    stable_flag = torch.hstack(stable_flag)

    # end simulation
    # for stop the solver
    for id in range(n_parallel):
        in_queue.put(None)

    # terminate all process
    for proc in jobs:
        proc.join()

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
    boundary = [0]
    default_settings['gurobi'] = {"nsim": 64}
    default_settings['env']['boundary_part_ids'] = boundary

    filename = os.path.join(RESOURCE_DIR, f"curriculum/{name}.pt")
    part_states = torch.load(filename)['input']

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
    # ipm_settings = ipm_init(parts, contacts, default_settings)
    # ipm_settings_cpu = ipm_update_device(ipm_settings, 'cpu')
    #
    # gpus = np.arange(torch.cuda.device_count())
    # devices = [f"cuda:{gpu_id}" for gpu_id in gpus]
    # simulators = ipm_init_simulate_parallel(ipm_settings_cpu, devices, n_batch)
    #
    # # search parameters
    # ipm_search_parameters_parallel(ipm_settings_cpu, part_states, simulators, 64, 0.9)
    #
    # # compute
    # n_batch = 512
    # sim_datas = ipm_split_states(ipm_settings_cpu, part_states, n_batch)
    # v_fp32, stable_fp32, avg_sim_time, avg_success_rate = ipm_simulate_parallel(sim_datas, simulators)
    # ipm_terminate(simulators)

    timer = perf_counter()
    init_rbe(parts, contacts, default_settings)
    v_fp32, stable_fp32 = gurobi_simulate_parallel(parts, contacts, part_states, default_settings)
    print("avg time", (perf_counter() - timer) / n_batch)
import os.path

import numpy as np
import warnings

import torch
from time import perf_counter
from scipy.cluster.vq import kmeans2
from trimesh import Trimesh
import time
from tqdm import tqdm
from learn2assemble.simulator import ipm_get_states, simulate
from learn2assemble.grasp import check_future_graspability
from learn2assemble.insertion import check_future_insertability, compute_insertion_table, compute_insertion_masks
import math

def cluster(part_states: np.ndarray,
            prev_inds: np.ndarray,
            ncluster: int):
    nbacth = part_states.shape[0]
    if ncluster >= nbacth:
        return part_states, prev_inds
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # shuffle
            arr = np.arange(nbacth)
            np.random.shuffle(arr)

            # bin_states = (part_states > 0).astype(float)
            bin_states = (part_states.copy()).astype(float)
            part_states = part_states[arr, :]
            bin_states, prev_inds = bin_states[arr, :], prev_inds[arr]

            # cluster
            centroids, labels = kmeans2(bin_states.astype(float), ncluster, minit='points')
            centroids = centroids.repeat(nbacth, axis=0)
            points = np.tile(bin_states, (ncluster, 1))
            dist = np.linalg.norm(points - centroids, axis=1).reshape(ncluster, nbacth)
            min = dist.argmin(axis=1)

            # return result
            return part_states[min, :], prev_inds[min]


def forward_install_actions(part_states: np.ndarray,
                            n_robot: int,
                            boundary_part_ids: list = []):
    # nbatch x naction
    nbatch = part_states.shape[0]
    naction = part_states.shape[1]
    prev_inds = np.arange(0, nbatch)
    prev_inds = np.tile(prev_inds, (naction, 1))
    prev_inds = prev_inds.swapaxes(0, 1)

    # nbatch x naction x npart
    new_states = part_states[:, None, :]
    new_states = np.tile(new_states, (1, naction, 1))

    # nbatch x naction x npart
    action = np.identity(naction, dtype=np.int32)
    action = np.tile(action, (nbatch, 1, 1))

    # not install an already installed parts
    flag = np.einsum("ijk, ijk->ij", new_states, action) == 0

    # not install on boundary
    flag[:, boundary_part_ids] = False

    # not use robots more than a given number
    flag_robot = (np.sum(new_states == 2, axis=2) + 1) <= n_robot + len(boundary_part_ids)

    flag = np.logical_and(flag, flag_robot)

    new_states = (new_states + action * 2)[flag, :]
    prev_inds = prev_inds[flag]

    return new_states, prev_inds


def forward_release_actions(part_states: np.ndarray,
                            boundary_part_ids: list = []):
    # nbatch x naction
    nbatch = part_states.shape[0]
    naction = part_states.shape[1]
    prev_inds = np.arange(0, nbatch)
    prev_inds = np.tile(prev_inds, (naction, 1))
    prev_inds = prev_inds.swapaxes(0, 1)

    # nbatch x naction x npart
    new_states = part_states[:, None, :]
    new_states = np.tile(new_states, (1, naction, 1))

    # nbatch x naction x npart
    action = np.identity(naction, dtype=np.int32)
    action = np.tile(action, (nbatch, 1, 1))

    # not install an already installed parts
    flag = np.einsum("ijk, ijk->ij", new_states == 2, action) == 1

    # not install on boundary
    flag[:, boundary_part_ids] = False

    new_states = (new_states - action)[flag, :]
    prev_inds = prev_inds[flag]

    return new_states, prev_inds


def forward_actions(part_states: np.ndarray,
                    n_robot: int,
                    boundary_part_ids: list = []):
    install_states, install_prev_inds = forward_install_actions(part_states, n_robot, boundary_part_ids)
    release_states, release_prev_inds = forward_release_actions(part_states, boundary_part_ids)

    new_states = np.vstack([install_states, release_states])
    prev_inds = np.hstack([install_prev_inds, release_prev_inds])
    release_action_flag = np.hstack([np.zeros(install_states.shape[0]), np.ones(release_states.shape[0])]).astype(
        np.bool_)

    new_states, unique_indices = np.unique(new_states, axis=0, return_index=True)
    prev_inds = prev_inds[unique_indices]
    release_action_flag = release_action_flag[unique_indices]
    install_action_flag = np.logical_not(release_action_flag)

    new_install_states = new_states[install_action_flag, :]
    new_release_states = new_states[release_action_flag, :]
    prev_install_states = part_states[prev_inds[install_action_flag], :]
    prev_release_states = part_states[prev_inds[release_action_flag], :]

    return new_install_states, prev_install_states, new_release_states, prev_release_states

def backward_actions(solution_dict: dict,
                     n_robot: int,
                     table_insertion,
                     boundary_part_ids: list = [],
                     *simulator):
    n_sim_buffer = 512
    part_states = []
    for key in solution_dict.keys():
        part_states.append(np.array(key, dtype = np.int32))
    part_states = np.vstack(part_states, dtype = np.int32)
    n_part = part_states.shape[1]
    labels = []
    n_bound = len(boundary_part_ids) + n_robot
    print("tot states", part_states.shape[0])
    # hold
    to_simulate_inds = []
    to_simulate_states = []
    for part_id in range(n_part):
        # new states
        label = np.ones(part_states.shape[0], dtype = np.int32) * (-2)
        if part_id not in boundary_part_ids:
            num_held = np.sum(part_states == 2, axis=1)
            flag = np.logical_and(num_held < n_bound, part_states[:, part_id] == 1)
            inds = np.arange(part_states.shape[0], dtype=np.int32)[flag]
            for ind in inds:
                backward_state = part_states[ind, :].copy()
                backward_state[part_id] = 2
                encode_state = tuple(backward_state)
                if encode_state in solution_dict:
                    label[ind] = 1
                # elif np.sum(backward_state == 2) == n_bound:
                #     to_simulate_inds.append([ind, part_id])
                #     held_ids = (backward_state == 2).nonzero()[0]
                #     for held_id in held_ids:
                #         if held_id not in boundary_part_ids:
                #             backward_state0 = backward_state.copy()
                #             backward_state0[held_id] = 0
                #             to_simulate_states.append(backward_state0)
                else:
                    label[ind] = 0
        labels.append(label.reshape(-1, 1))

    labels_held = np.hstack(labels)
    if len(to_simulate_states) > 0:
        to_simulate_inds = np.array(to_simulate_inds)
        to_simulate_states = np.vstack(to_simulate_states)
        flag, _ = simulate_buffer(to_simulate_states, n_sim_buffer, boundary_part_ids, *simulator)
        flag = np.any(flag.reshape(-1, 2), axis = 1)
        to_simulate_inds = to_simulate_inds[flag, :]
        labels_held[to_simulate_inds[:, 0], to_simulate_inds[:, 1]] = 0

    # remove
    to_simulate_inds = []
    to_simulate_states = []
    labels = []
    if table_insertion is not None:
        insertion_masks = compute_insertion_masks(part_states, boundary_part_ids, table_insertion)
    else:
        insertion_masks = np.ones(part_states.shape, dtype=np.bool_)
    for part_id in range(n_part):
        label = np.ones(part_states.shape[0], dtype = np.int32) * (-2)
        if part_id not in boundary_part_ids:
            flag = (part_states[:, part_id] == 2)
            # new states
            inds = np.arange(part_states.shape[0], dtype=np.int32)[flag]
            for ind in inds:
                backward_state = part_states[ind, :].copy()
                backward_state[part_id] = 0
                encode_state = tuple(backward_state)
                if insertion_masks[ind, part_id] == False:
                    label[ind] = -2
                elif encode_state in solution_dict:
                    label[ind] = 1
                else:
                    label[ind] = -1
                    to_simulate_inds.append([ind, part_id])
                    to_simulate_states.append(backward_state)
        labels.append(label.reshape(-1, 1))
    # remove
    labels_remove = np.hstack(labels)
    if len(to_simulate_states) > 0:
        to_simulate_inds = np.array(to_simulate_inds)
        to_simulate_states = np.vstack(to_simulate_states)
        flag, n_test = simulate_buffer(to_simulate_states, n_sim_buffer, boundary_part_ids, *simulator)
        print("remove check:\t", f"{np.sum(flag.astype(np.int32))} / {n_test}")
        to_simulate_inds = to_simulate_inds[flag, :]
        labels_remove[to_simulate_inds[:, 0], to_simulate_inds[:, 1]] = 0

    # remove
    labels = np.hstack([labels_held, labels_remove])
    print("-2", np.sum(labels == -2))
    print("-1", np.sum(labels == -1))
    print("0", np.sum(labels == 0))
    print("1", np.sum(labels == 1))
    return labels

def check_terminate(part_states: np.ndarray,
                    boundary_part_ids: list = []):
    npart = part_states.shape[1]
    fixed_states = np.ones(npart)
    fixed_states[boundary_part_ids] = 2
    dist = np.sum(np.abs(part_states - fixed_states[None, :]), axis=1)
    return dist == 0


def compute_solution(n_part, boundary_part_ids, solution_dict):
    # append complete state to the end
    part_state = np.ones(n_part, dtype=np.int32)
    part_state[boundary_part_ids] = 2
    part_state_encode = tuple(part_state.tolist())
    solution = [part_state]
    while part_state_encode in solution_dict:
        if len(solution_dict[part_state_encode]) > 0:
            part_state_encode = solution_dict[part_state_encode][0]
            solution.append(np.array(part_state_encode))
        else:
            break
    solution.reverse()
    return np.array(solution, dtype=np.int32)

def list_vstack(array_list, n_part):
    if len(array_list) == 0:
        return np.zeros((0, n_part))
    else:
        return np.vstack(array_list)

def array_stack(array0, array1):
    if array0.shape[0] == 0:
        return array1
    elif array1.shape[0] == 0:
        return array0
    else:
        return np.vstack([array0, array1])

def add_solution(curr_states, prev_states, solution_dict, n_part):
    new_curr_states = []
    for id in range(curr_states.shape[0]):
        curr_state_encode = tuple(curr_states[id].tolist())
        prev_state_encode = tuple(prev_states[id].tolist())
        if curr_state_encode not in solution_dict:
            solution_dict[curr_state_encode] = [prev_state_encode]
            new_curr_states.append(curr_states[id])
        else:
            solution_dict[curr_state_encode].append(prev_state_encode)
    return list_vstack(new_curr_states, n_part)

def remove_duplicated_simulation(part_states, solution_dict, n_part):
    _, inds = np.unique(part_states, return_index=True, axis=0)
    new_curr_states = []
    new_inds = []
    for id in inds:
        curr_state_encode = tuple(part_states[id].tolist())
        if curr_state_encode not in solution_dict:
            new_curr_states.append(part_states[id])
            new_inds.append(id)
    return list_vstack(new_curr_states, n_part), np.array(new_inds)

def simulate_buffer(part_states, n_buffer, boundary_part_ids, *simulator):
    flag = np.zeros(part_states.shape[0], dtype = np.bool_)
    it = 0
    states = torch.tensor(part_states.copy(), dtype = torch.int32)
    n_step = int(math.ceil(states.shape[0] / n_buffer))
    if n_step > 2:
        progress = tqdm(total=n_step, position=1)
    else:
        progress = None
    while states.shape[0] > 0:
        if states.shape[0] < n_buffer and not simulator[3]:
            break
        test_states, n_test = ipm_get_states(states, boundary_part_ids, n_buffer)
        _, test_flag = simulate(simulator[0], simulator[1], test_states, simulator[2])
        flag[it: it + n_test] = test_flag[:n_test].numpy()
        states = states[n_test:, :]
        it += n_test
        if progress is not None:
            progress.update()
    return flag[:it], it

def forward_curriculum(parts: list[Trimesh],
                       contacts: list[dict],
                       table_insertion=None,
                       table_grasp=None,
                       settings: dict = {}):
    # parameters
    env = settings.get("env", {})
    boundary_part_ids = env.get("boundary_part_ids", [])
    n_robot = env.get("n_robot", 2)

    curriculum_settings = update_default_settings(settings, "curriculum", {
        "n_beam": 64,
        "buffer_size": 1E6,
        "verbose": False,
        "n_sim_batch": 1024})

    n_beam = curriculum_settings["n_beam"]
    verbose = curriculum_settings["verbose"]
    n_sim_batch = curriculum_settings["n_sim_batch"]
    buffer_size = int(curriculum_settings["buffer_size"])
    # init states
    states_to_expand = np.zeros((1, len(parts)), dtype=np.int32)
    states_to_expand[:, boundary_part_ids] = 2

    # beam search
    curriculum = []
    records = {
        "part_states": [],
        "prev_inds": [],
    }

    policy_dataset = {
        "input": [],
        "output": []
    }
    n_part = len(parts)
    states_stack = np.zeros((0, n_part), dtype=np.int32)
    states_to_simulate = np.zeros((0, n_part), dtype=np.int32)
    prev_states_to_simulate = np.zeros((0, n_part), dtype=np.int32)
    solution_dict = {tuple(states_to_expand[0].tolist()): []}
    complete_state = np.ones(n_part, dtype=np.int32)
    complete_state[boundary_part_ids] = 2
    encode_complete_state = tuple(complete_state.tolist())

    while (states_to_expand.shape[0] > 0):
        install_states, prev_install_states, release_states, prev_release_states = forward_actions(states_to_expand,
                                                                                                   n_robot,
                                                                                                      boundary_part_ids)

        # install
        if install_states.shape[0] > 0:
            # check insertion
            if table_insertion is not None:
                insertability = check_future_insertability(install_states, table_insertion)
                install_states = install_states[insertability, :]
                prev_install_states = prev_install_states[insertability, :]

            # check grasp
            if prev_install_states.shape[0] > 0 and table_grasp is not None:
                graspability_flag = check_future_graspability(install_states, boundary_part_ids, table_grasp)
                install_states = install_states[graspability_flag, :]
                prev_install_states = prev_install_states[graspability_flag]

            if prev_install_states.shape[0] > 0:
                new_states = add_solution(install_states, prev_install_states, solution_dict, n_part)
                states_stack = array_stack(states_stack, new_states)

        # remove
        states_to_simulate = array_stack(states_to_simulate, release_states)
        prev_states_to_simulate = array_stack(prev_states_to_simulate, prev_release_states)
        if states_to_simulate.shape[0] > 0:
            states_to_simulate, inds = remove_duplicated_simulation(states_to_simulate, solution_dict, n_part)
            prev_states_to_simulate = prev_states_to_simulate[inds, :]

            complete_simulation = (states_stack.shape[0] == 0)
            timer = perf_counter()
            flag, n_test = simulate_buffer(states_to_simulate,
                                           n_sim_batch,
                                           boundary_part_ids,
                                           parts,
                                           contacts,
                                           settings, complete_simulation)
            if n_test > 0:
                test_states = states_to_simulate[:n_test, :]
                test_prev_states = prev_states_to_simulate[:n_test, :]
                if verbose:
                    max_part = np.max(np.sum(test_states >= 1, axis=1))
                    print("max_part:\t", max_part,
                          ",\t sim:\t", f"{np.sum(flag)}/{n_test}",
                          ",\t time:\t", round((time.perf_counter() - timer) / n_test, 4))

                new_states = add_solution(test_states[flag, :], test_prev_states[flag, :], solution_dict, n_part)
                states_stack = array_stack(states_stack, new_states)

                states_to_simulate = states_to_simulate[n_test:, :]
                prev_states_to_simulate = prev_states_to_simulate[n_test:, :]

        if encode_complete_state in solution_dict:
            break

        if states_stack.shape[0] > 0:
            # remove states
            if states_stack.shape[0] > buffer_size:
                states_stack = states_stack[-buffer_size: -1, :]

            # only sample states with the max number of parts
            num_parts = np.sum(states_stack >= 1, axis=1)
            max_part = np.max(num_parts)
            weights = np.ones(states_stack.shape[0], dtype=np.float32)
            weights[num_parts < max_part] = 0.0
            num_state = int(np.sum(weights))
            weights = weights / np.sum(weights)
            sample_inds = np.random.choice(
                np.arange(states_stack.shape[0]),
                size=min(n_beam, num_state),
                replace=False,  # Key parameter to ensure no duplicates
                p=weights
            )
            states_to_expand = states_stack[sample_inds, :]

            # remove sample from queue
            flag = np.ones(states_stack.shape[0], dtype=bool)
            flag[sample_inds] = False
            states_stack = states_stack[flag, :]
        else:
            return False, solution_dict
    else:
        return False, solution_dict
    return True, solution_dict


if __name__ == '__main__':
    from learn2assemble import ASSEMBLY_RESOURCE_DIR, update_default_settings, default_settings, RESOURCE_DIR
    from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
    from learn2assemble.render import render_sequence, init_polyscope
    import polyscope as ps
    import polyscope.imgui as psim

    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + "/tetris-1")
    default_settings['curriculum']['verbose'] = True
    default_settings['rbe']['velocity_tol'] = 1E-2
    default_settings['rbe']['mu'] = 0.2
    #default_settings['gurobi'] = {}
    default_settings["assembly"]["contact_shrink_ratio"] = 0.1  # for robustnessly computing the contact surfaces
    default_settings['curriculum']['n_beam'] = 64
    default_settings['curriculum']['n_sim_batch'] = 512
    default_settings['insertion']['type'] = 'planar'
    # default_settings['env']['boundary_part_ids'] = [len(parts) - 1]
    # default_settings['ipm']["n_pcg_iter"] = 200

    contacts = compute_assembly_contacts(parts, default_settings)
    table_insertion, drts = compute_insertion_table(parts, default_settings)
    # table_grasp, grasp_frames, _ = compute_grasp_table(parts, default_settings)
    succeed, solution_dict = forward_curriculum(parts, contacts, table_insertion, None, default_settings)
    print(succeed)
    torch.save(solution_dict, ASSEMBLY_RESOURCE_DIR + "/solution.pt")
    solution_dict = torch.load(ASSEMBLY_RESOURCE_DIR + "/solution.pt")
    labels = backward_actions(solution_dict, 2, table_insertion, default_settings['env']['boundary_part_ids'], parts, contacts, default_settings, True)
    solution = compute_solution(len(parts), default_settings['env']['boundary_part_ids'], solution_dict)
    if solution is not None:
        init_polyscope()
        render_sequence(parts, solution, default_settings, True)
        ps.show()

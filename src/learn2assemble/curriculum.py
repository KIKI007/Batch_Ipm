import os.path

import numpy as np
import warnings

import torch
from scipy.cluster.vq import kmeans2
from trimesh import Trimesh
import time

from learn2assemble.simulator import ipm_get_states, simulate
from learn2assemble.grasp import check_future_graspability
from learn2assemble.insertion import check_future_insertability


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


def check_terminate(part_states: np.ndarray,
                    boundary_part_ids: list = []):
    npart = part_states.shape[1]
    fixed_states = np.ones(npart)
    fixed_states[boundary_part_ids] = 2
    dist = np.sum(np.abs(part_states - fixed_states[None, :]), axis=1)
    return dist == 0

def compute_solution(part_state, solution_dict):
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


def compute_policy_labels(current_states, prev_states):
    n_part = current_states.shape[1]
    policy_labels = np.zeros((current_states.shape[0], 2 * n_part), dtype=int)

    # hold
    for part_id in range(n_part):
        update_current_states = np.copy(current_states)
        update_current_states[:, part_id] = 2
        x = np.arange(current_states.shape[0])
        y = np.arange(prev_states.shape[0])
        xv, yv = np.meshgrid(x, y, indexing='ij')
        diff = np.linalg.norm(update_current_states[xv, :] - prev_states[yv, :], axis = 2)
        flag = (diff < 1E-6).any(axis = 1)
        policy_labels[:, part_id] = np.logical_and(flag, current_states[:, part_id] == 1)

    # remove
    for part_id in range(n_part):
        update_current_states = np.copy(current_states)
        update_current_states[:, part_id] = 0
        x = np.arange(current_states.shape[0])
        y = np.arange(prev_states.shape[0])
        xv, yv = np.meshgrid(x, y, indexing='ij')
        diff = np.linalg.norm(update_current_states[xv, :] - prev_states[yv, :], axis = 2)
        flag = (diff < 1E-6).any(axis = 1)
        policy_labels[:, n_part + part_id] = np.logical_and(flag, current_states[:, part_id] == 2)

    return policy_labels

def add_to_map(states, prev_states, solution_dict):
    for id, state in enumerate(states):
        state_encode = tuple(state.tolist())
        prev_state_encode = tuple(prev_states[id].tolist())
        if state_encode not in solution_dict:
            solution_dict[state_encode] = [prev_state_encode]
        else:
            solution_dict[state_encode].append(prev_state_encode)

def array_stack(array0, array1):
    if array0.shape[0] == 0:
        return array1
    elif array1.shape[0] == 0:
        return array0
    else:
        return np.vstack([array0, array1])

def forward_curriculum(parts: list[Trimesh],
                       contacts: list[dict],
                       table_insertion = None,
                       table_grasp=None,
                       settings: dict = {}):

    # parameters
    env = settings.get("env", {})
    boundary_part_ids = env.get("boundary_part_ids", [])
    n_robot = env.get("n_robot", 2)

    curriculum_settings = update_default_settings(settings, "curriculum", {"n_beam": 64, "verbose": False, "n_sim_batch": 2048})

    n_beam = curriculum_settings["n_beam"]
    verbose = curriculum_settings["verbose"]
    n_sim_batch =  curriculum_settings["n_sim_batch"]

    # init states
    part_states = np.zeros((1, len(parts)), dtype=np.int32)
    part_states[:, boundary_part_ids] = 2
    iter = -1

    # beam search
    curriculum = []
    records = {
        "part_states": [],
        "prev_inds": [],
    }

    policy_dataset = {
        "input":  [],
        "output": []
    }

    states_to_explore = np.zeros((0, len(parts)), dtype=np.int32)
    prev_states_to_explore = np.zeros((0, len(parts)), dtype=np.int32)
    states_to_simulate = np.zeros((0, len(parts)), dtype=np.int32)
    prev_states_to_simulate = np.zeros((0, len(parts)), dtype=np.int32)
    solution_dict = {}
    max_part = 1
    while (part_states.shape[0] > 0):
        iter += 1
        install_states, prev_install_states, release_states, prev_release_states = forward_actions(part_states, n_robot, boundary_part_ids)
        states_to_simulate = array_stack(states_to_simulate, release_states)
        prev_states_to_simulate = array_stack(prev_states_to_simulate, prev_release_states)
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
                states_to_explore = array_stack(states_to_explore, install_states)
                prev_states_to_explore = array_stack(prev_states_to_explore, prev_install_states)

        while states_to_simulate.shape[0] >= n_sim_batch or (states_to_simulate.shape[0] > 0 and states_to_explore.shape[0] == 0):
            timer = time.perf_counter()
            if n_sim_batch < states_to_simulate.shape[0]:
                test_states = torch.tensor(states_to_simulate[:n_sim_batch, :], dtype=torch.long, device='cpu')
                n_test = n_sim_batch
            else:
                test_states = torch.tensor(states_to_simulate, dtype=torch.long, device='cpu')
                test_states, n_test = ipm_get_states(test_states, boundary_part_ids, n_sim_batch)
            _, flag = simulate(parts, contacts, test_states, settings)
            flag = flag[:n_test].numpy().astype(bool)
            test_states = test_states[:n_test, :].numpy()
            test_prev_states = prev_states_to_simulate[:n_test, :]
            if verbose:
                print("max_part:\t", max_part,
                      ",\t sim:\t", f"{np.sum(flag)}/{n_test}",
                      ",\t time:\t", round((time.perf_counter() - timer) / n_test, 4))

            states_to_explore = array_stack(states_to_explore, test_states[flag, :])
            prev_states_to_explore = array_stack(prev_states_to_explore, test_prev_states[flag, :])

            states_to_simulate = states_to_simulate[n_test:, :]
            prev_states_to_simulate = prev_states_to_simulate[n_test:]

        if states_to_explore.shape[0] > 0:
            # remove states
            num_parts = np.sum(states_to_explore >= 1, axis = 1)
            max_part = np.max(num_parts)
            flag = np.ones(states_to_explore.shape[0], dtype=bool)
            flag[num_parts + 2 < max_part] = False
            states_to_explore = states_to_explore[flag, :]
            prev_states_to_explore = prev_states_to_explore[flag, :]

            # sample states
            n_sample = min(n_beam, states_to_explore.shape[0])
            weights = num_parts[flag]
            weights = weights / np.sum(weights)
            sampled_inds = np.random.choice(
                np.arange(states_to_explore.shape[0]),
                size=n_sample,
                replace=False,  # Key parameter to ensure no duplicates
                p=weights
            )
            part_states = states_to_explore[sampled_inds, :]
            prev_part_states = prev_states_to_explore[sampled_inds, :]
            add_to_map(part_states, prev_part_states, solution_dict)
            curriculum.append(part_states)
            if check_terminate(part_states, boundary_part_ids).any():
                break

            # update states
            flag = np.ones(states_to_explore.shape[0], dtype=bool)
            flag[sampled_inds] = False
            states_to_explore = states_to_explore[flag, :]
            prev_states_to_explore = prev_states_to_explore[flag, :]
        else:
            curriculum = np.vstack(curriculum)
            return False, compute_solution(curriculum[-1, :], solution_dict), curriculum

    # append complete state to the end
    complete_state = np.ones(len(parts), dtype=np.int32)
    complete_state[boundary_part_ids] = 2
    curriculum.append(complete_state)
    return True, compute_solution(complete_state, solution_dict), np.vstack(curriculum)


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
    default_settings["assembly"]["contact_shrink_ratio"] = 0.1 # for robustnessly computing the contact surfaces
    default_settings['curriculum']['n_beam'] = 64
    #default_settings['env']['boundary_part_ids'] = [len(parts) - 1]
    #default_settings['ipm']["n_pcg_iter"] = 200

    contacts = compute_assembly_contacts(parts, default_settings)
    #table_insertion, drts = compute_insertion_table(parts, default_settings)
    #table_grasp, grasp_frames, _ = compute_grasp_table(parts, default_settings)
    succeed, solution, curriculum = forward_curriculum(parts, contacts, None, None, default_settings)
    print("succeed:\t", succeed)

    init_polyscope()
    render_sequence(parts, solution, default_settings)
    ps.show()

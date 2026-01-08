import os.path

import numpy as np
import warnings

import torch
from scipy.cluster.vq import kmeans2
from trimesh import Trimesh
import time

from learn2assemble.simulator import ipm_get_states, simulate
from learn2assemble.grasp import check_future_graspability
from learn2assemble.insertion import check_future_insertability, compute_insertion_table


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

    part_states = []
    for key in solution_dict.keys():
        part_states.append(np.array(key, dtype = np.int32))
    part_states = np.vstack(part_states, dtype = np.int32)
    n_part = part_states.shape[1]
    labels = []
    n_bound = len(boundary_part_ids) + n_robot

    # hold
    to_simulate_inds = []
    to_simulate_states = []
    for part_id in range(n_part):
        num_held = np.sum(part_states == 2, axis=1)
        flag = np.logical_and(num_held < n_bound, part_states[:, part_id] == 1)

        # new states
        label = np.ones(part_states.shape[0], dtype = np.int32) * (-1)
        inds = np.arange(part_states.shape[0], dtype=np.int32)[flag]

        #
        for ind in inds:
            backward_state = part_states[ind, :].copy()
            backward_state[part_id] = 2
            encode_state = tuple(backward_state)
            if encode_state in solution_dict:
                label[ind] = 1
            elif np.sum(backward_state == 2) == n_bound:
                to_simulate_inds.append([part_id, ind])
                held_ids = (backward_state[part_id] == 2).nonzero()[0]
                for held_id in held_ids:
                    backward_state0 = backward_state.copy()
                    backward_state0[held_id] = 0
                    to_simulate_states.append(backward_state0)
            else:
                label[ind] = 0

        labels.append(label.reshape(-1, 1))

    labels = np.hstack(labels)
    print(labels)
    to_simulate_inds = np.array(to_simulate_inds)
    to_simulate_states = np.vstack(to_simulate_states)
    _, flag = simulate(simulator[0], simulator[1], to_simulate_states, simulator[2])
    flag = np.any(flag.reshape(-1, 2), axis = 1)
    labels[to_simulate_inds[flag]] = 0
    print(labels)

    # remove
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
        diff = np.linalg.norm(update_current_states[xv, :] - prev_states[yv, :], axis=2)
        flag = (diff < 1E-6).any(axis=1)
        policy_labels[:, part_id] = np.logical_and(flag, current_states[:, part_id] == 1)

    # remove
    for part_id in range(n_part):
        update_current_states = np.copy(current_states)
        update_current_states[:, part_id] = 0
        x = np.arange(current_states.shape[0])
        y = np.arange(prev_states.shape[0])
        xv, yv = np.meshgrid(x, y, indexing='ij')
        diff = np.linalg.norm(update_current_states[xv, :] - prev_states[yv, :], axis=2)
        flag = (diff < 1E-6).any(axis=1)
        policy_labels[:, n_part + part_id] = np.logical_and(flag, current_states[:, part_id] == 2)

    return policy_labels


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
    new_prev_states = []
    for id in range(curr_states.shape[0]):
        curr_state_encode = tuple(curr_states[id].tolist())
        prev_state_encode = tuple(prev_states[id].tolist())
        if curr_state_encode not in solution_dict:
            solution_dict[curr_state_encode] = [prev_state_encode]
            new_curr_states.append(curr_states[id])
            new_prev_states.append(prev_states[id])
        else:
            solution_dict[curr_state_encode].append(prev_state_encode)
    return list_vstack(new_curr_states, n_part), list_vstack(new_prev_states, n_part)


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
    buffer_size = curriculum_settings["buffer_size"]
    # init states
    states_to_expand = np.zeros((1, len(parts)), dtype=np.int32)
    states_to_expand[:, boundary_part_ids] = 2
    iter = -1

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
    prev_states_stack = np.zeros((0, n_part), dtype=np.int32)
    states_to_simulate = np.zeros((0, n_part), dtype=np.int32)
    prev_states_to_simulate = np.zeros((0, n_part), dtype=np.int32)
    solution_dict = {}

    while (states_to_expand.shape[0] > 0):
        iter += 1
        install_states, prev_install_states, release_states, prev_release_states = forward_actions(states_to_expand,
                                                                                                   n_robot,
                                                                                                   boundary_part_ids)
        states_to_simulate = array_stack(states_to_simulate, release_states)
        prev_states_to_simulate = array_stack(prev_states_to_simulate, prev_release_states)

        new_states = np.zeros((0, n_part), dtype=np.int32)
        prev_new_states = np.zeros((0, n_part), dtype=np.int32)
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
                new_states = array_stack(new_states, install_states)
                prev_new_states = array_stack(prev_new_states, prev_install_states)

        while states_to_simulate.shape[0] >= n_sim_batch or (
                states_to_simulate.shape[0] > 0 and states_stack.shape[0] == 0):
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
                max_part = np.max(np.sum(test_states, axis=1))
                print("max_part:\t", max_part,
                      ",\t sim:\t", f"{np.sum(flag)}/{n_test}",
                      ",\t time:\t", round((time.perf_counter() - timer) / n_test, 4))

            new_states = array_stack(new_states, test_states[flag, :])
            prev_new_states = array_stack(prev_new_states, test_prev_states[flag, :])

            states_to_simulate = states_to_simulate[n_test:, :]
            prev_states_to_simulate = prev_states_to_simulate[n_test:]

        new_states, prev_new_states = add_solution(new_states, prev_new_states, solution_dict, n_part)
        if check_terminate(new_states, boundary_part_ids).any():
            break

        states_stack = array_stack(states_stack, new_states)
        prev_states_stack = array_stack(prev_states_stack, prev_new_states)
        if states_stack.shape[0] > 0:
            # remove states
            if states_stack.shape[0] > buffer_size:
                states_stack = states_stack[-buffer_size:, :]
                prev_states_stack = prev_states_stack[-buffer_size:, :]

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
            if verbose:
                max_part = np.max(np.sum(states_stack, axis=1))
                print("max_part:\t", max_part,
                      ",\t num stack:\t", f"{states_stack.shape[0]}")

            # remove sample from queue
            flag = np.ones(states_stack.shape[0], dtype=bool)
            flag[sample_inds] = False
            states_stack = states_stack[flag, :]
            prev_states_stack = prev_states_stack[flag, :]
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
    default_settings['gurobi'] = {}
    default_settings["assembly"]["contact_shrink_ratio"] = 0.1  # for robustnessly computing the contact surfaces
    default_settings['curriculum']['n_beam'] = 64
    default_settings['curriculum']['n_sim_batch'] = 512
    # default_settings['env']['boundary_part_ids'] = [len(parts) - 1]
    # default_settings['ipm']["n_pcg_iter"] = 200

    contacts = compute_assembly_contacts(parts, default_settings)
    table_insertion, drts = compute_insertion_table(parts, default_settings)
    # table_grasp, grasp_frames, _ = compute_grasp_table(parts, default_settings)
    #succeed, solution_dict = forward_curriculum(parts, contacts, table_insertion, None, default_settings)
    #torch.save(solution_dict, os.path.join(RESOURCE_DIR, "solution.pt"))
    solution_dict = torch.load(os.path.join(RESOURCE_DIR, "solution.pt"))
    backward_actions(solution_dict, 2, table_insertion, default_settings['env']['boundary_part_ids'], parts, contacts, default_settings)
    solution = compute_solution(len(parts), default_settings['env']['boundary_part_ids'], solution_dict)
    print("succeed:\t", True)

    if solution is not None:
        init_polyscope()
        render_sequence(parts, solution, default_settings)
        ps.show()

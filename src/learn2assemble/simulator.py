import copy
from time import perf_counter
import gurobipy as gp
import torch
from gurobipy import GRB
from learn2assemble.rbe import *
from types import SimpleNamespace
import platform
from learn2assemble.rbe import num_vars
import torch.multiprocessing as mp

logger = {
    'timer': {},
    'log': {},
    'activate': True,
}

if platform.system() == 'Windows' or platform.system() == 'Darwin':
    disable_compile = True
else:
    disable_compile = False


def inf_norm(x):
    return torch.max(torch.abs(x), dim=0).values

def reset_timer(name):
    if logger['activate']:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if name not in logger['timer']:
            logger['timer'][name] = perf_counter()
            logger['log'][name] = 0.0
        else:
            logger['timer'][name] = perf_counter()

def end_timer(name):
    if logger['activate']:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if 'log' not in logger:
            logger['log'] = {}
        if name in logger['timer']:
            logger['log'][name] += perf_counter() - logger['timer'][name]


def print_logger(nbatch=1.0, names=[]):
    if logger['activate']:
        if len(names) == 0:
            for name, value in logger['log'].items():
                print(name, f":\t\t\t {value / nbatch:.3e}")
        else:
            for name in names:
                if name in logger['log']:
                    print(name, f":\t\t\t {logger['log'][name] / nbatch:.3e}")

@torch.compile(disable=disable_compile)
def GT_(p, nλn, nt, nf, mu):
    nλt = nλn * nt
    nx = nλn + nλt + nf
    λ0 = torch.zeros((nx, p.shape[1]), device=p.device, dtype=p.dtype)
    λ0[:nλn, :] = mu * (- p[:nλn])
    λ0[nλn: nλn + nλt, :] = p[:nλn].repeat((nt, 1))
    λ2 = p[nλn: nλn + nx]
    λ3 = p[nλn + nx:]
    return λ3 - λ2 + λ0

@torch.compile(disable=disable_compile)
def G_(p, nλn, nt, nf, mu):
    nλt = nλn * nt
    λn, λt = p[:nλn, :], p[nλn: nλn + nλt, :]
    inds = torch.arange(nλn, device=p.device, dtype=torch.long)
    inds = inds.repeat(nt)
    λn = mu * λn
    λn.index_add_(0, inds, λt, alpha=-1.0)
    return torch.vstack([-λn, -p, p])

@torch.compile(disable=disable_compile)
def GTZSG_(p, ZS, nλn, nt, nf, mu):
    nbatch = p.shape[1]
    nλt = nλn * nt
    nx = nλn + nλt + nf

    # step 1
    λn, λt = p[:nλn, :], p[nλn: nλn + nλt, :]
    inds = torch.arange(nλn, device=p.device, dtype=torch.long)
    inds = inds.repeat(nt)
    λn = mu * λn
    λn.index_add_(0, inds, λt, alpha=-1.0)
    Gp = torch.vstack([-λn, -p, p])

    # step 2
    ZSGp = ZS * Gp

    # step 3
    λ0 = torch.zeros((nx, nbatch), device=p.device, dtype=p.dtype)
    λ0[:nλn, :] = mu * (- ZSGp[:nλn])
    λ0[nλn: nλn + nλt, :] = ZSGp[:nλn].repeat((nt, 1))
    λ2 = ZSGp[nλn: nλn + nx]
    λ3 = ZSGp[nλn + nx:]
    return λ3 - λ2 + λ0

@torch.compile(disable=disable_compile)
def Q_(p, nλn, nt, iA, iB, nA, nB, invM, Q):
    batch = p.shape[1]
    nλt = nλn * nt
    λ, f = p[: nλn + nλt, :], -p[nλn + nλt:, :]
    λ = λ[:, None, :].repeat(1, 6, 1).reshape(-1, batch)
    xA = λ * nA[:, None]
    xB = λ * nB[:, None]
    f.index_add_(0, iA, xA)
    f.index_add_(0, iB, xB)
    f = f.reshape(-1, 3, batch)
    x = (invM @ f).reshape(-1, batch)
    pA = torch.index_select(x, 0, iA)
    pB = torch.index_select(x, 0, iB)
    λ = nA[:, None] * pA + nB[:, None] * pB
    λ = torch.sum(λ.reshape(-1, 6, batch), dim=1)
    return torch.vstack([λ, -x])

def ipm_contacts(ipm, parts, contacts, density, boundary_part_ids):
    iAs = []
    iBs = []
    nAs = []
    nBs = []

    device = ipm["device"]
    float_type = ipm["float_type"]
    nt = ipm['nt']
    ipm['nf'], ipm['nλn'], ipm['nλt'] = num_vars(parts, contacts, nt)

    # contacts
    for contact in contacts:
        iA = contact["iA"]
        iB = contact["iB"]
        nrm = contact["plane"][:3, 2]
        for part_id in [iB, iA]:
            ct = parts[part_id].center_mass
            inds = np.arange(part_id * 6, part_id * 6 + 6)
            for id, pt in enumerate(contact["mesh"].vertices):
                m = np.hstack([nrm, np.cross(pt - ct, nrm)])
                if part_id is iB:
                    iBs.append(inds)
                    nBs.append(m)
                else:
                    iAs.append(inds)
                    nAs.append(m)
            nrm = -nrm

    for k in range(nt):
        angle = 2.0 * math.pi / nt * k
        for contact in contacts:
            iA = contact["iA"]
            iB = contact["iB"]
            xaxis = contact["plane"][:3, 0]
            yaxis = contact["plane"][:3, 1]
            t = xaxis * np.cos(angle) + yaxis * np.sin(angle)
            for part_id in [iB, iA]:
                ct = parts[part_id].center_mass
                inds = np.arange(part_id * 6, part_id * 6 + 6)
                for id, pt in enumerate(contact["mesh"].vertices):
                    m = np.hstack([t, np.cross(pt - ct, t)])
                    if part_id is iB:
                        iBs.append(inds)
                        nBs.append(m)
                    else:
                        iAs.append(inds)
                        nAs.append(m)
                t = -t

    iAs = np.hstack(iAs)
    iBs = np.hstack(iBs)
    nAs = np.hstack(nAs)
    nBs = np.hstack(nBs)
    ipm['iAs'] = torch.tensor(iAs, device=device, dtype=torch.long)
    ipm['iBs'] = torch.tensor(iBs, device=device, dtype=torch.long)
    ipm['nAs'] = torch.tensor(nAs, device=device, dtype=float_type)
    ipm['nBs'] = torch.tensor(nBs, device=device, dtype=float_type)

    # mass and gravity
    volumes, moment_inertias = compute_volume_and_inertial(parts, density, boundary_part_ids)
    M = []
    ipm['g'] = torch.zeros(ipm['nf'], device=device, dtype=float_type)
    for part_id, part in enumerate(parts):
        # mass
        Mi = np.identity(3) * volumes[part_id] * density
        Ii = moment_inertias[part_id] * density
        M.append(np.linalg.inv(Mi))
        M.append(np.linalg.inv(Ii))

        # gravity
        if part_id not in boundary_part_ids:
            ipm['g'][part_id * 6 + 2] = -volumes[part_id] * density

    M = np.stack(M)
    ipm['invM'] = torch.tensor(M, device=device, dtype=ipm['float_type'])

def ipm_init(parts: list[Trimesh],
             contacts: list[dict],
             settings: dict):

    if platform.system() == "Darwin":
        default_device = "mps"
    elif torch.cuda.is_available():
        default_device = "cuda"
    else:
        default_device = "cpu"

    update_default_settings(settings,
                            "ipm",
                            {
                                "n_iter": 30,
                                "n_pcg_eval_iter": 10,
                                "n_linesearch": 32,
                                "kkt_conv_eps": 1E-5,
                                "pcg_rel_eps": 1E-2,
                                "x_bound_tol": 1E-6,
                                "float_type": torch.float32,
                                "device": torch.device(default_device)})

    ipm = update_default_settings(settings, "ipm", settings["rbe"])

    device = ipm["device"]
    float_type = ipm['float_type']

    ipm['n_part'] = len(parts)
    ipm['boundary_part_ids'] = settings['env']['boundary_part_ids']
    ipm['density'] = compute_best_density(parts, ipm['boundary_part_ids'])
    ipm_contacts(ipm, parts, contacts, ipm['density'], ipm['boundary_part_ids'])

    # set low precision multiple
    torch.set_float32_matmul_precision('high')

    # compute pre-conditioner
    nx = ipm['nλn'] * (ipm["nt"] + 1) + ipm["nf"]
    rbeG = ipm["nλn"], ipm["nt"], ipm["nf"], ipm["mu"]
    rbeQ = ipm['nλn'], ipm['nt'], ipm['iAs'], ipm['iBs'], ipm['nAs'], ipm['nBs'], ipm['invM'], None
    p = torch.eye(nx, device=device, dtype=float_type)
    G = G_(p, *rbeG)
    ipm['GG'] = G * G
    ipm['Q'] = Q_(p, *rbeQ)
    ipm['diagQ'] = torch.diagonal(ipm['Q'])

    H = GT_(G, *rbeG) + ipm['Q']
    if p.device.type == "mps":
        cholesky_H = torch.linalg.cholesky(H.to(device = 'cpu').to(dtype = torch.float64))
        ipm['cholesky_H'] = torch.cholesky_inverse(cholesky_H).to(device = device, dtype=float_type)
        ipm['invH'] = torch.cholesky_inverse(cholesky_H).to(device = device, dtype=float_type)
    else:
        cholesky_H = torch.linalg.cholesky(H)
        ipm['invH'] = torch.cholesky_inverse(cholesky_H)
        ipm['cholesky_H'] = cholesky_H

    # auto parameters
    settings["ipm"]["Ccp"] = 1.2 * abs(torch.sum(settings['ipm']['g']).item())
    if "n_pcg_iter" not in settings["ipm"]:
        settings["ipm"]["n_pcg_iter"] = max(int(100), int((settings["ipm"]["Q"].shape[0] * 0.03) // 10 * 10))
    print("density = ", settings["ipm"]["density"])
    print("Ccp = ", settings["ipm"]["Ccp"])
    print("num pcg iter = ", settings["ipm"]["n_pcg_iter"])

    settings['ipm'] = ipm
    ipm['pre-computed'] = True
    return ipm

def ipm_empty_states(n_part, bounary_part_ids, n_batch):
    padding = torch.zeros((n_batch, n_part), device="cpu", dtype=torch.long)
    padding[:, bounary_part_ids] = 2
    return padding

def ipm_sort_states(part_states, ascend = True):
    inds = torch.sum((part_states >= 1), dim=1).cpu().numpy()
    inds = np.argsort(inds).tolist()
    if not ascend:
        inds = inds[::-1]
    inds = torch.tensor(inds, device=part_states.device, dtype=torch.long)
    return part_states[inds, :]

def ipm_get_states(part_states, boundary_part_ids, n_sample):
    if n_sample <= 0:
        return None
    if (n_sample & (n_sample - 1)) != 0:
        n_sample = int(2 ** np.ceil(np.log2(n_sample)))
        print("change num sample to ", n_sample)

    test_states = ipm_empty_states(part_states.shape[1], boundary_part_ids, n_sample)
    test_states_sub = part_states[: min(part_states.shape[0], n_sample):, :]
    n_test_sub = test_states_sub.shape[0]
    test_states[:n_test_sub, :] = test_states_sub
    return test_states, n_test_sub

def ipm_search_parameters(ipm_settings: dict, part_states, nsample = 32, acc_tol=0.9):
    new_states = ipm_sort_states(part_states, ascend=False)
    test_states, n_test_sub = ipm_get_states(new_states, ipm_settings['boundary_part_ids'], n_sample = nsample)
    n_pcg_it = ipm_settings["n_pcg_iter"]
    best_acc = 0.0
    for scale in [1, 1.5, 2, 2.5, 3, 3.5, 4]:
        ipm_settings["n_pcg_iter"] = int(n_pcg_it * scale)
        _, flag = ipm_simulate(test_states, ipm_settings)
        flag = flag[:n_test_sub]
        acc = torch.sum(flag).item() / flag.shape[0]
        best_acc = max(best_acc, acc)
        print("num pcg iter = ", ipm_settings["n_pcg_iter"], f" with a {best_acc: .2f} success rate")
        if best_acc > acc_tol:
            return True
    else:
        ipm_settings["n_pcg_iter"] = 400
        print(f"Failed to find pcg iter with a maximum {best_acc: .2f} success rate")
        return False

def ipm_index_mapping(batch_part_states, iAs, iBs, nλn, device):
    if batch_part_states.ndim == 1:
        batch_part_states = batch_part_states.reshape(1, -1)

    if torch.is_tensor(batch_part_states):
        part_states = batch_part_states.clone().to(device=device, dtype=torch.long)
    else:
        part_states = torch.tensor(batch_part_states, device=device, dtype=torch.long)

    n_batch = part_states.shape[0]
    p = (part_states == 1)[:, :, None].repeat(1, 1, 6).reshape(n_batch, -1)
    p = p.T
    A = (iAs.reshape(-1, 6)[:nλn, 0] // 6).type(torch.long)
    B = (iBs.reshape(-1, 6)[:nλn, 0] // 6).type(torch.long)
    pA = torch.index_select(part_states, 1, A)
    pB = torch.index_select(part_states, 1, B)
    flag0 = torch.logical_and(pA > 0, pB > 0)
    flag1 = torch.logical_or(pA < 2, pB < 2)
    c = torch.logical_and(flag0, flag1).T
    return p, c

def ipm_get_dynamic_attrib(ipm, batch_part_states):
    nλn = ipm.nλn
    nλt = ipm.nλt
    Ccp = ipm.Ccp
    nt = ipm.nt
    device = ipm.device

    ps, cs = ipm_index_mapping(batch_part_states, ipm.iAs, ipm.iBs, ipm.nλn, device=device)
    ps, cs = ps.type(ipm.float_type), cs.type(ipm.float_type)

    n_batch = batch_part_states.shape[0]
    Pg = ps * ipm.g[:, None]

    # invM
    Pg = Pg.reshape(-1, 3, n_batch)
    Pg = (ipm.invM @ Pg).reshape(-1, n_batch)

    # KT
    pA = torch.index_select(Pg, 0, ipm.iAs)
    pB = torch.index_select(Pg, 0, ipm.iBs)
    λ = ipm.nAs[:, None] * pA + ipm.nBs[:, None] * pB
    λ = torch.sum(λ.reshape(-1, 6, n_batch), dim=1)
    q = torch.vstack([λ, -Pg])

    Al = torch.zeros((ipm.nλn, n_batch), dtype=ipm.float_type, device=device)
    xl = torch.vstack([torch.zeros((nλn, n_batch), dtype=ipm.float_type, device=device),
                       torch.zeros((nλt, n_batch), dtype=ipm.float_type, device=device),
                       -Ccp * (1 - ps)])

    xu = torch.vstack([Ccp * cs,
                       torch.tile(Ccp * cs, (nt, 1)),
                       Ccp * (1 - ps)])

    return q, xl, xu, Al, ps, cs

def ipm_sum_forces(p, nλn, nt, iA, iB, nA, nB):
    batch = p.shape[1]
    nλt = nλn * nt
    nf = p.shape[0] - nλn - nλt
    λ = p[: nλn + nλt, :]
    λ = λ[:, None, :].repeat(1, 6, 1).reshape(-1, batch)
    xA = λ * nA[:, None]
    xB = λ * nB[:, None]
    sumf = torch.zeros((nf, p.shape[1]), device=p.device, dtype=p.dtype)
    sumf.index_add_(0, iA, xA)
    sumf.index_add_(0, iB, xB)
    return sumf

def ipm_evaluate_result(ipm, xclip, ps):
    batch = xclip.shape[1]
    rbe = ipm.nλn, ipm.nt, ipm.iAs, ipm.iBs, ipm.nAs, ipm.nBs
    residual = (ipm_sum_forces(xclip, *rbe) + ipm.g[:, None]) * ps
    residual = residual.reshape(-1, 3, batch)
    velocity = (ipm.invM @ residual).reshape(-1, batch)
    velocity_inf_nrm = inf_norm(velocity)
    return velocity, velocity_inf_nrm

def ipm_start_solve(ipm, h, q):
    rbe = ipm.nλn, ipm.nt, ipm.nf, ipm.mu
    b = GT_(h, *rbe) - q
    if h.device.type == 'mps':
        x = ipm.invH @ b # for mac
    else:
        x = torch.cholesky_solve(b, ipm.cholesky_H)

    oldz = G_(x, *rbe) - h
    alpha_p = torch.max(oldz, 0).values
    flag = (alpha_p < 0).type(h.dtype).repeat(oldz.shape[0], 1)
    s = flag * (-oldz) + (1 - flag) * (-oldz + (1 + alpha_p))

    alpha_d = -torch.min(oldz, 0).values
    flag = (alpha_d < 0).type(h.dtype).repeat(oldz.shape[0], 1)
    z = flag * (oldz) + (1 - flag) * (oldz + 1 + alpha_d)
    return x, s, z

def ipm_kkt_res(ipm, q, h, x, s, z):
    rbe = ipm.nλn, ipm.nt, ipm.nf, ipm.mu
    rbeQ = ipm.nλn, ipm.nt, ipm.iAs, ipm.iBs, ipm.nAs, ipm.nBs, ipm.invM, ipm.Q
    r1 = Q_(x, *rbeQ) + q + GT_(z, *rbe)
    r2 = s * z
    r3 = G_(x, *rbe) + s - h
    kkt_res = inf_norm(torch.vstack([r1, r2, r3]))
    return r1, r2, r3, kkt_res

def ipm_precond(ipm, s, z):
    ZS = z / s
    diagG = torch.einsum("ji, jb -> ib", ipm.GG, ZS)
    invM = diagG + ipm.diagQ[:, None]
    invM = 1.0 / invM
    return invM

@torch.compile(disable=disable_compile)
def ipm_linesearch(s, ds, z, dz, n_sample=32):
    device = s.device
    alpha = torch.linspace(0, 1, n_sample, device=device, dtype=s.dtype)
    ls = s[None, :, :] + alpha[:, None, None] * ds[None, :, :]
    lz = z[None, :, :] + alpha[:, None, None] * dz[None, :, :]
    flag = torch.logical_and((ls >= 0).all(dim=1), (lz >= 0).all(dim=1)).type(s.dtype)
    inds = torch.arange(n_sample, device=device, dtype=s.dtype)
    inds = flag * inds[:, None]
    ind = torch.max(inds, dim=0).values.to(torch.long)
    return alpha[ind]

@torch.compile(disable=disable_compile)
def ipm_centering_params(s, z, ds_a, dz_a, n_sample):
    """duality gap + cc term in predictor-corrector PDIP"""
    sz = torch.sum(s * z, dim=0)
    mu = sz / s.shape[0]
    alpha = ipm_linesearch(s, ds_a, z, dz_a, n_sample=n_sample)
    sigma = torch.sum((s + alpha * ds_a) * (z + alpha * dz_a), dim=0) / sz
    sigma = sigma ** 3
    return sigma, mu

@torch.compile(disable=disable_compile)
def ipm_solve_rhs(ipm, s, z, invP, v1, v2, v3, n_iter, dx=None):
    device = ipm.device

    rbeG = ipm.nλn, ipm.nt, ipm.nf, ipm.mu
    rbeQ = ipm.nλn, ipm.nt, ipm.iAs, ipm.iBs, ipm.nAs, ipm.nBs, ipm.invM, ipm.Q

    ZS = z / s
    b = GT_((z * v3 - v2) / s, *rbeG) + v1

    if dx is None:
        dx = torch.zeros_like(b)
        xk = torch.zeros_like(b)
        rk = b.clone()
    else:
        xk = dx.clone()
        rk = b - (GTZSG_(dx, ZS, *rbeG) + Q_(dx, *rbeQ))
        dx = torch.zeros_like(b)

    dx_rk = torch.ones(b.shape[1], device=device, dtype=b.dtype) * 1E9

    uk = invP * rk
    pk = uk.clone()

    m = n_iter // ipm.n_pcg_eval_iter
    # reset_timer('pcg')
    for k in range(m):
        for t in range(ipm.n_pcg_eval_iter):
            # Apk = GT @ (ZS * (G @ pk)) + Q @ pk
            # fast computation
            Apk = GTZSG_(pk, ZS, *rbeG) + Q_(pk, *rbeQ)

            ru = torch.sum(rk * uk, dim=0)
            ak = ru / torch.sum(pk * Apk, dim=0)
            xk += ak[None, :] * pk
            rk -= ak[None, :] * Apk
            uk = invP * rk
            betak = torch.sum(rk * uk, dim=0) / ru
            pk = uk + betak[None, :] * pk

        # only update x when rk decrease
        #prev_rk = dx_rk.clone()
        error = inf_norm(rk)
        flag = error < dx_rk
        dx[:, flag] = xk[:, flag]
        dx_rk[flag] = error[flag]
        # rel = torch.max(torch.abs(dx_rk - prev_rk) / prev_rk)

        # recompute the residual to avoid numerical errors
        rk = b - (GTZSG_(xk, ZS, *rbeG) + Q_(xk, *rbeQ))
        uk = invP * rk
        # if rel < ipm.rel_eps:
        #     break
        abs_ = torch.max(dx_rk)
        if abs_ < ipm.kkt_conv_eps / 10:
            break

    ds = v3 - G_(dx, *rbeG)
    dz = (v2 - z * ds) / s
    return dx, ds, dz

def ipm_update_device(ipm, device):
    new_ipm = {}
    for name, val in ipm.items():
        if torch.is_tensor(val):
            new_ipm[name] = val.clone().to(device).share_memory_()
        else:
            new_ipm[name] = copy.deepcopy(val)
    new_ipm["device"] = torch.device(device)
    return new_ipm

def ipm_simulate(batch_part_states: list[dict], ipm_settings):
    # name space
    reset_timer('ipm')
    ipm = SimpleNamespace(**ipm_settings)
    floatType = ipm.float_type
    device = ipm.device

    # update dynamic attributes
    reset_timer('dynamic_attrib')
    q, xl, xu, Al, ps, cs = ipm_get_dynamic_attrib(ipm, batch_part_states)
    xl, xu = xl - ipm.x_bound_tol, xu + ipm.x_bound_tol
    h = torch.vstack([-Al, -xl, xu])
    end_timer('dynamic_attrib')

    # initialize ipm x0
    reset_timer('start_solve')
    x, s, z = ipm_start_solve(ipm, h, q)
    result_x = torch.zeros_like(x)
    kkt_res_best = torch.ones(batch_part_states.shape[0], device=device, dtype=floatType) * 1E9
    r1, r2, r3, kkt_res = ipm_kkt_res(ipm, q, h, x, s, z)
    end_timer('start_solve')

    # ipm main loop
    for it in range(ipm.n_iter):

        # 1. pre-conditioner
        reset_timer('pre-cond')
        invP = ipm_precond(ipm, s, z)
        end_timer('pre-cond')

        # 2. solve kkt 1
        reset_timer('kkt 1')
        dx_a, ds_a, dz_a = ipm_solve_rhs(ipm, s, z, invP, -r1, -r2, -r3, n_iter=ipm.n_pcg_iter)
        end_timer('kkt 1')

        # 3. centering parameters
        reset_timer('center')
        sigma, mu = ipm_centering_params(s, z, ds_a, dz_a, n_sample=ipm.n_linesearch)
        end_timer('center')

        # 4. solve kkt 2
        reset_timer('kkt 2')

        # option 1
        # r2 -= (sigma * mu - (ds_a * dz_a))
        # dx, ds, dz = ipm_solve_rhs(Q, s, z, invM, -r1, -r2, -r3, n_iter = pcg_iter, dx = dx_a, rbe = rbe)

        # option 2
        r2 = (sigma * mu - (ds_a * dz_a))
        dx, ds, dz = ipm_solve_rhs(ipm, s, z, invP, 0, r2, 0, n_iter=ipm.n_pcg_iter // 2)
        dx, ds, dz = dx + dx_a, ds + ds_a, dz + dz_a
        end_timer('kkt 2')

        # 5. step size and update
        reset_timer('linesearch')
        alpha = 0.99 * ipm_linesearch(s, ds, z, dz, n_sample=ipm.n_linesearch)
        end_timer('linesearch')

        reset_timer('update')
        # update x
        x = x + alpha * dx
        s = s + alpha * ds
        z = z + alpha * dz

        # update result_x based on kkt residual
        pre_res = kkt_res_best.clone()
        r1, r2, r3, kkt_res = ipm_kkt_res(ipm, q, h, x, s, z)
        flag = kkt_res_best > kkt_res
        kkt_res_best[flag] = kkt_res[flag]
        result_x[:, flag] = x[:, flag]
        kkt_res_best = torch.clip(kkt_res_best, ipm.kkt_conv_eps, torch.inf)

        # # remove converged
        # flag = kkt_res > ipm.kkt_conv_eps
        # inds = inds[flag]
        # q, h, x, s, z, invP = q[:, flag], h[:, flag], x[:, flag], s[:, flag], z[:, flag], invP[:, flag]
        # r1, r2, r3 = r1[:, flag], r2[:, flag], r3[:, flag]

        #rel_ = torch.max(torch.abs(kkt_res_best - pre_res) / pre_res)
        abs_ = torch.max(kkt_res_best)
        #print(rel_, abs_)
        if abs_ < ipm.kkt_conv_eps:
            break
        end_timer('update')

    xclip = torch.clip(result_x, xl, xu)
    velocity, velocity_inf_nrm = ipm_evaluate_result(ipm, xclip, ps)
    end_timer('ipm')
    return velocity.cpu(), (velocity_inf_nrm < ipm.velocity_tol).cpu()

def ipm_simulate_parallel_proc(job_id, queue): #ipm_settings, return_dict):
    torch.set_float32_matmul_precision('high')
    #velocity, stable_flag = ipm_simulate(part_states, ipm_settings)
    part_states = torch.zeros((10, 10), device = 'cuda:0', dtype = torch.long)
    part_states = part_states.share_memory_()
    queue.put((job_id, part_states))

def ipm_simulate_parallel(batch_part_states: torch.tensor, list_ipm_settings):
    n_parallel = len(list_ipm_settings)
    n_state_per_process = batch_part_states.shape[0] // n_parallel

    streams = []
    return_dict = []
    batch_part_states = batch_part_states.to(device = 'cpu')
    for id in range(n_parallel):
        ipm_settings = list_ipm_settings[id]
        device = torch.device(ipm_settings['device'])
        s = torch.cuda.Stream(device=device)
        streams.append(s)
        if id != n_parallel - 1:
            inds = torch.arange(id * n_state_per_process,
                                n_state_per_process * (id + 1),
                                device='cpu',
                                dtype=torch.long)
        else:
            # last take all
            inds = torch.arange(id * n_state_per_process,
                                batch_part_states.shape[0],
                                device='cpu',
                                dtype=torch.long)
        part_states = batch_part_states[inds, :]
        with torch.cuda.stream(s):
            part_states = part_states.to(device = device, non_blocking=True)
            return_dict.append(ipm_simulate(part_states, ipm_settings))

    velocity = []
    stable_flag = []
    for id in range(n_parallel):
        streams[id].synchronize()
        velocity.append(return_dict[id][0])
        stable_flag.append(return_dict[id][1])

    velocity = torch.hstack(velocity)
    stable_flag = torch.hstack(stable_flag)

    return velocity, stable_flag

def init_gurobi(parts, contacts, settings: dict):
    params = {
        "WLSACCESSID": "9d6cfee4-4a06-46b1-a7c8-a7445b4e62a6",
        "WLSSECRET": "563345f3-3017-488e-a549-eb6742256f41",
        "LICENSEID": 2759892,
        "OptimalityTol": 1E-6,
        "OutputFlag": settings["rbe"]["verbose"],
        "Method": -1,
    }
    env = gp.Env(params=params)
    settings["gurobi"] = {
        "env": env,
        "pre-computed": True
    }

def simulate_gurobi(batch_part_states: list[dict],
                    settings: dict):
    reset_timer('gurobi')
    env = settings["gurobi"]["env"]

    # rbe
    rbe = SimpleNamespace(**settings["rbe"])

    ps, cs = rbe.mapping(batch_part_states)
    q, xl, xu, Al, Au = compute_rbe_dynamic_attribs(rbe.g, ps, cs, rbe.Jn, rbe.Jt, rbe.invM, rbe.mu, rbe.Ccp)

    flags, vs = [], []
    nx, nA = q.shape[0], Al.shape[0]

    for id in range(batch_part_states.shape[0]):
        xli, xui, Ali, Aui, qi = xl[:, id], xu[:, id], Al[:, id], Au[:, id], q[:, id]
        m = gp.Model(env=env)
        x = m.addMVar(nx, lb=xli, ub=xui)
        y = m.addMVar(rbe.L.shape[0], lb=-GRB.INFINITY, ub=GRB.INFINITY)

        m.setObjective(0.5 * y @ y + qi @ x, gp.GRB.MINIMIZE)
        m.addConstr(rbe.L @ x[:rbe.L.shape[1]] == y)
        m.addConstr(rbe.A @ x <= Aui)
        m.addConstr(rbe.A @ x >= Ali)
        m.optimize()

        if m.Status == GRB.OPTIMAL:
            xclip = np.clip(x.X, xli, xui)
            λn, λt = xclip[:rbe.nλn], xclip[rbe.nλn: rbe.nλn + rbe.nλt]
            residual = (rbe.Jn.T @ λn + rbe.Jt.T @ λt + rbe.g) * ps[:, id]
            velocity = rbe.invM @ residual
            velocity_inf_nrm = np.max(np.abs(velocity), axis=0)
            # print(velocity_inf_nrm)
            if velocity_inf_nrm < rbe.velocity_tol:
                flags.append(True)
            else:
                flags.append(False)
            vs.append(velocity)
        else:
            flags.append(False)
            vs.append(np.zeros(rbe.nf))
    vs = torch.tensor(vs, device ="cpu", dtype=torch.float32)
    vs = vs.T
    flags = torch.tensor(flags, device ="cpu", dtype=torch.bool)
    end_timer('gurobi')
    return vs, flags

def simulate(parts: list[Trimesh],
             contacts: list[dict],
             batch_part_states: list[dict],
             settings: dict):
    if "gurobi" in settings:
        rbe_pre_computed = settings.get("rbe", {"pre-computed": False}).get("pre-computed", False)
        if not rbe_pre_computed:
            init_rbe(parts, contacts, settings)
        gurobi_pre_computed = settings["gurobi"].get("pre-computed", False)
        if not gurobi_pre_computed:
            init_gurobi(parts, contacts, settings)
        return simulate_gurobi(batch_part_states, settings)
    else:
        ipm_computed = settings.get("ipm", {"pre-computed": False}).get("pre-computed", False)
        if not ipm_computed:
            ipm_init(parts, contacts, settings)
        x, flag = ipm_simulate(batch_part_states, settings["ipm"])
        return x, flag

if __name__ == '__main__':
    from learn2assemble import ASSEMBLY_RESOURCE_DIR, default_settings, RESOURCE_DIR
    from learn2assemble.render import *
    from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
    import os

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        exit(0)

    default_settings['rbe']['mu'] = 0.2
    default_settings["assembly"]["contact_shrink_ratio"] = 0.1  # for robustnessly computing the contact surfaces

    n_batch = 2048
    torch.manual_seed(0)
    name = "tetris-999"
    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + f"/{name}")
    default_settings['env']['boundary_part_ids'] = [len(parts) - 1]

    filename = os.path.join(RESOURCE_DIR, f"curriculum/{name}.pt")
    part_states = torch.load(filename)['input']
    #part_states[:, 10] = 0
    #part_states[:, 31] = 0

    # choose the max parts
    inds = torch.sum(part_states, dim=1).cpu().numpy()
    inds = np.argsort(inds).tolist()[::-1]
    part_states = part_states[inds, :]

    # random
    part_states = part_states[:n_batch, :]

    # default_settings['gurobi'] = {}
    default_settings['ipm'] = {
        "n_iter": 25,
        "n_pcg_iter": 200,
        "n_pcg_eval_iter": 10,
        "x_bound_tol": 1E-5,
        "kkt_conv_eps": 1E-4,
        "float_type": torch.float32,
    }
    logger['activate'] = False

    reset_timer('contact')
    contacts = compute_assembly_contacts(parts, default_settings)
    end_timer('contact')

    reset_timer('init ipm')
    ipm_settings = ipm_init(parts, contacts, default_settings)
    end_timer('init ipm')

    ipm_settings = [ipm_update_device(ipm_settings, 'cuda:0')]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timer = perf_counter()

    #devices = ["cuda:0", "cuda:1"]
    # devices = ["cuda:0"]
    v_fp32, stable_fp32 = ipm_simulate_parallel(part_states, ipm_settings)
    #v_fp32, stable_fp32 = simulate(parts, contacts, part_states, default_settings)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sim_time = perf_counter() - timer

    print("time ", sim_time / stable_fp32.shape[0])
    print(torch.sum(stable_fp32).item() / stable_fp32.shape[0])
    print_logger(1)

    # #render
    # import polyscope as ps
    #
    # init_polyscope()
    # t = 0
    #
    # def callback():
    #     global t
    #     changed, t = psim.SliderFloat("time", v=t, v_min=0, v_max=1)
    #     if changed:
    #         draw_assembly_motion(parts, part_states[0], v_fp32[:, 0] * t)
    #
    #
    # draw_contacts(contacts, part_states[0])
    # draw_assembly_motion(parts, part_states[0], v_fp32[:, 0] * t)
    # ps.set_user_callback(callback)
    # ps.show()

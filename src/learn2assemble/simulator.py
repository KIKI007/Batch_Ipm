import math
from time import perf_counter

import scipy as sp
from trimesh import Trimesh
import torch
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from learn2assemble.rbe import *
from types import SimpleNamespace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = {
    'timer' : {},
    'log': {},
    'activate': True,
}

def from_scipy_to_torch_sparse(A: sp.sparse.coo_matrix,
                               floatType=torch.float32):
    return torch.sparse_coo_tensor(torch.LongTensor(np.vstack((A.row, A.col))),
                                   torch.tensor(A.data, dtype=floatType),
                                   torch.Size(A.shape)).to(device)


def inf_norm(x):
    return torch.max(torch.abs(x), dim=0).values

def evaluate_result(xclip, ps, rbe):
    λn, λt = xclip[:rbe.nλn, :], xclip[rbe.nλn: rbe.nλn + rbe.nλt, :]
    residual = (rbe.JnT @ λn + rbe.JtT @ λt + rbe.g[:, None]) * ps
    velocity = rbe.invMass @ residual
    velocity_inf_nrm = inf_norm(velocity)
    return velocity, velocity_inf_nrm

def reset_timer(name):
    if logger['activate']:
        torch.cuda.synchronize()
        if name not in logger['timer']:
            logger['timer'][name] = perf_counter()
            logger['log'][name] = 0.0
        else:
            logger['timer'][name] = perf_counter()

def end_timer(name):
    if logger['activate']:
        torch.cuda.synchronize()
        if 'log' not in logger:
            logger['log'] = {}
        if name in logger['timer']:
            logger['log'][name] += perf_counter() - logger['timer'][name]

def print_logger(nbatch = 1.0):
    if logger['activate']:
        for name, value in logger['log'].items():
            print(name, f":\t\t\t {value / nbatch:.3e}")

def get_dynamic_attrib(batch_part_states, rbe, float_type):
    ps, cs = rbe.mapping(batch_part_states)
    q, xl, xu, Al, Au = compute_rbe_dynamic_attribs(rbe.g, ps, cs, rbe.Jn, rbe.Jt, rbe.invM, rbe.mu, rbe.Ccp)
    q = torch.tensor(q, dtype=float_type, device=device)
    xl = torch.tensor(xl, dtype=float_type, device=device)
    xu = torch.tensor(xu, dtype=float_type, device=device)
    Al = torch.tensor(Al, dtype=float_type, device=device)
    Au = torch.tensor(Au, dtype=float_type, device=device)
    return q, xl, xu, Al, Au, ps, cs

def ipm_contacts(ipm, parts, contacts, density, boundary_part_ids):

    iAs = []
    iBs = []
    nAs = []
    nBs = []
    nt = ipm['nt']

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
    ipm['nAs'] = torch.tensor(nAs, device=device, dtype=ipm['float_type'])
    ipm['nBs'] = torch.tensor(nBs, device=device, dtype=ipm['float_type'])

    # invM
    volumes, moment_inertias = compute_volume_and_inertial(parts, density, boundary_part_ids)
    M = []
    for part_id, part in enumerate(parts):
        Mi = np.identity(3) * volumes[part_id] * density
        Ii = moment_inertias[part_id] * density
        M.append(np.linalg.inv(Mi))
        M.append(np.linalg.inv(Ii))
    M = np.stack(M)
    ipm['invM'] = torch.tensor(M, device=device, dtype=ipm['float_type'])

def init_ipm(parts: list[Trimesh], contacts: list[dict], settings: dict):
    ipm = update_default_settings(settings,
                                  "ipm",
                                  {
                                      "n_iter": 30,
                                      "n_pcg_iter_1": 100,
                                      "n_pcg_iter_2": 50,
                                      "n_pcg_eval_iter": 20,
                                      "n_linesearch": 32,
                                      "kkt_conv_eps": 1E-5,
                                      "x_bound_tol": 1E-6,
                                      "float_type": torch.float32,
                                  })
    float_type = ipm['float_type']
    rbe = settings['rbe']
    A = rbe["A"]
    L = rbe["L"]
    Q = L.T @ L

    nx = A.shape[1]
    Inx = sp.sparse.coo_matrix(sp.sparse.eye_array(nx, dtype=np.float64))

    G = sp.sparse.block_array([[-A],
                               [-Inx],
                               [Inx]])
    GG = G * G

    ipm['Q'] = torch.tensor(Q.todense(), dtype=float_type, device=device)
    ipm['diagQ'] = torch.tensor(np.diagonal(Q.todense()), dtype=float_type, device=device)
    ipm['GG'] = torch.tensor(GG.todense(), dtype=float_type, device=device)

    H = Q + G.T @ G
    H_tch = torch.tensor(H.todense(), dtype=torch.float64, device=device)
    cholesky_H = torch.linalg.cholesky(H_tch)
    ipm['invH'] = torch.cholesky_inverse(cholesky_H).type(float_type)

    # variables from rbe
    ipm['nλn'] = rbe['nλn']
    ipm['nλt'] = rbe['nλt']
    ipm['nf'] = rbe['nf']
    ipm['nt'] = rbe['nt']
    ipm['mu'] = rbe['mu']
    ipm['g'] = torch.tensor(rbe['g'], dtype=float_type, device=device)
    ipm['JnT'] = torch.tensor(rbe['Jn'].todense().transpose(), dtype=float_type, device=device)
    ipm['JtT'] = torch.tensor(rbe['Jt'].todense().transpose(), dtype=float_type, device=device)
    ipm['invMass'] = torch.tensor(rbe['invM'].todense(), dtype=float_type, device=device)

    Inf = torch.eye(ipm['nf'], device=device, dtype= float_type)
    ipm['KnT'] = torch.hstack([ipm['JnT'], ipm['JtT'], -Inf])
    ipm_contacts(ipm, parts, contacts, rbe['density'], settings['env']['boundary_part_ids'])

    ipm['pre-computed'] = True
    settings['ipm'] = ipm

#@torch.compile
def GT_(p, nλn, nt, nf, mu):
    nλt = nλn * nt
    nx = nλn + nλt + nf
    λ0 = torch.zeros((nx, p.shape[1]), device=p.device, dtype=p.dtype)
    λ0[:nλn, :] = mu * (- p[:nλn])
    λ0[nλn: nλn + nλt, :] = p[:nλn].repeat((nt, 1))
    λ2 = p[nλn: nλn + nx]
    λ3 = p[nλn + nx:]
    return λ3 - λ2 + λ0

#@torch.compile
def G_(p, nλn, nt, nf, mu):
    nλt = nλn * nt
    λn, λt = p[:nλn, :], p[nλn: nλn + nλt, :]
    inds = torch.arange(nλn, device=p.device, dtype=torch.long)
    inds = inds.repeat(nt)
    λn = mu * λn
    λn.index_add_(0, inds, λt, alpha=-1.0)
    return torch.vstack([-λn, -p, p])

#@torch.compile
def GTZSG(p, ZS, nλn, nt, nf, mu):
    Gp = G_(p, nλn, nt, nf, mu)
    ZSGp = ZS * Gp
    return GT_(ZSGp, nλn, nt, nf, mu)

# def Knt_(p, nλn, nt, iA, iB, nA, nB, invM):
#     nλt = nλn * nt
#     λ, f = p[: nλn + nλt, :], -p[nλn + nλt: , :]
#     xA = torch.einsum('ib, ij -> ijb', λ, nA).reshape(-1, p.shape[1])
#     xB = torch.einsum('ib, ij -> ijb', λ, nB).reshape(-1, p.shape[1])
#     f.index_add_(0, iA, xA)
#     f.index_add_(0, iB, xB)
#     return f
#
# def Kn_(p, nλn, nt, iA, iB, nA, nB, invM):
#     nbatch = p.shape[1]
#     pA = torch.index_select(p, 0, iA).reshape(-1, 6, nbatch)
#     pB = torch.index_select(p, 0, iB).reshape(-1, 6, nbatch)
#     λ = nA[:, :, None] * pA + nB[:, :, None] * pB
#     λ = torch.sum(λ, dim = 1)
#     return torch.vstack([λ, -p])
#
# def invM_(p, nλn, nt, iA, iB, nA, nB, invM):
#     b = p.shape[1]
#     p_ = p.reshape(-1, 3, b)
#     x = torch.einsum('ijk, ikb -> ijb', invM, p_).reshape(-1, b)
#     return x

@torch.compile
def Q_(p, nλn, nt, iA, iB, nA, nB, invM):
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

@torch.compile
def ipm_start_solve(ipm, h, q):
    rbe = ipm.nλn, ipm.nt, ipm.nf, ipm.mu
    x = ipm.invH @ (GT_(h, *rbe) - q)
    oldz = G_(x, *rbe) - h
    alpha_p = torch.max(oldz, 0).values
    flag = (alpha_p < 0).type(h.dtype).repeat(oldz.shape[0], 1)
    s = flag * (-oldz) + (1 - flag) * (-oldz + (1 + alpha_p))

    alpha_d = -torch.min(oldz, 0).values
    flag = (alpha_d < 0).type(h.dtype).repeat(oldz.shape[0], 1)
    z = flag * (oldz) + (1 - flag) * (oldz + 1 + alpha_d)
    return x, s, z

@torch.compile
def ipm_kkt_res(ipm, q, h, x, s, z):
    rbe = ipm.nλn, ipm.nt, ipm.nf, ipm.mu
    rbeQ = ipm.nλn, ipm.nt, ipm.iAs, ipm.iBs, ipm.nAs, ipm.nBs, ipm.invM
    r1 = Q_(x, *rbeQ) + q + GT_(z, *rbe)
    r2 = s * z
    r3 = G_(x, *rbe) + s - h
    kkt_res = inf_norm(torch.vstack([r1, r2, r3]))
    return r1, r2, r3, kkt_res

@torch.compile
def ipm_precond(ipm, s, z):
    ZS = z / s
    GTZSG = torch.einsum("ji, jb -> ib", ipm.GG, ZS)
    invM = GTZSG + ipm.diagQ[:, None]
    invM = 1.0 / invM
    return invM

@torch.compile
def ipm_solve_rhs(ipm, s, z, invP, v1, v2, v3, n_iter, dx=None):
    rbe = ipm.nλn, ipm.nt, ipm.nf, ipm.mu
    rbeQ = ipm.nλn, ipm.nt, ipm.iAs, ipm.iBs, ipm.nAs, ipm.nBs, ipm.invM

    ZS = z / s
    b = GT_((z * v3 - v2) / s, *rbe) + v1

    if dx is None:
        dx = torch.zeros_like(b)
        xk = torch.zeros_like(b)
        rk = b.clone()
    else:
        xk = dx.clone()
        rk = b - (GTZSG(dx, ZS, *rbe) + Q_(dx, *rbeQ))
        dx = torch.zeros_like(b)

    dx_rk = torch.ones(b.shape[1], device=device, dtype=b.dtype) * torch.inf

    uk = invP * rk
    pk = uk.clone()

    m = n_iter // ipm.n_pcg_eval_iter
    #reset_timer('pcg')
    for k in range(m):
        for t in range(ipm.n_pcg_eval_iter):
            # Apk = GT @ (ZS * (G @ pk)) + Q @ pk
            # fast computation
            Apk = GTZSG(pk, ZS, *rbe) + Q_(pk, *rbeQ)

            ru = torch.sum(rk * uk, dim=0)
            ak = ru / torch.sum(pk * Apk, dim=0)
            xk += ak[None, :] * pk
            rk -= ak[None, :] * Apk
            uk = invP * rk
            betak = torch.sum(rk * uk, dim=0) / ru
            pk = uk + betak[None, :] * pk

        # only update x when rk decrease
        error = inf_norm(rk)
        flag = error < dx_rk
        dx[:, flag] = xk[:, flag]
        dx_rk[flag] = error[flag]

        # recompute the residual to avoid numerical errors
        rk = b - (GTZSG(xk, ZS, *rbe) + Q_(xk, *rbeQ))
        uk = invP * rk
    #end_timer('pcg')

    ds = v3 - G_(dx, *rbe)
    dz = (v2 - z * ds) / s
    return dx, ds, dz

@torch.compile
def linesearch(s, ds, z, dz, n_sample=32):
    alpha = torch.linspace(0, 1, n_sample, device=device, dtype=s.dtype)
    ls = s[None, :] + torch.einsum('i, jk -> ijk', alpha, ds)
    lz = z[None, :] + torch.einsum('i, jk -> ijk', alpha, dz)
    flag = torch.logical_and((ls >= 0).all(dim=1), (lz >= 0).all(dim=1))
    inds = torch.arange(n_sample, device=device, dtype=s.dtype)
    inds = torch.einsum('ib, i -> ib', flag, inds)
    ind = torch.max(inds, dim=0).values.to(torch.long)
    return alpha[ind]

@torch.compile
def centering_params(s, z, ds_a, dz_a, n_sample):
    """duality gap + cc term in predictor-corrector PDIP"""
    sz = torch.sum(s * z, dim=0)
    mu = sz / s.shape[0]
    alpha = linesearch(s, ds_a, z, dz_a, n_sample=n_sample)
    sigma = torch.sum((s + alpha * ds_a) * (z + alpha * dz_a), dim=0) / sz
    sigma = sigma ** 3
    return sigma, mu

def simulate_ipm(batch_part_states: list[dict],
                 settings: dict):
    # name space
    rbe = SimpleNamespace(**settings["rbe"])
    ipm = SimpleNamespace(**settings["ipm"])
    floatType = ipm.float_type

    q, xl, xu, Al, Au, ps, cs = get_dynamic_attrib(batch_part_states, rbe, ipm.float_type)
    xl, xu = xl - ipm.x_bound_tol, xu + ipm.x_bound_tol
    h = torch.vstack([-Al, -xl, xu])
    ps = torch.tensor(ps, dtype=floatType, device=device)

    # initialize ipm x0
    x, s, z = ipm_start_solve(ipm, h, q)
    result_x = torch.zeros_like(x)
    inds = torch.arange(s.shape[1], dtype=torch.long, device=device)
    kkt_res_best = torch.ones(part_states.shape[0], device=device, dtype=floatType) * torch.inf
    r1, r2, r3, kkt_res = ipm_kkt_res(ipm, q, h, x, s, z)

    # rbeQ = ipm.nλn, ipm.nt, ipm.iAs, ipm.iBs, ipm.nAs, ipm.nBs, ipm.invM
    # #
    # Qx = ipm.Q @ x
    # Qx_new = Q_(x, *rbeQ)
    # print(torch.linalg.norm(Qx - Qx_new))
    #
    # y = x.clone()
    # reset_timer('Qx')
    # for it in range(1000):
    #     Qx = ipm.Q @ y
    #     y = y + 1
    # end_timer('Qx')
    #
    # y = x.clone()
    # reset_timer('Qx_')
    # for it in range(1000):
    #     Qx = Q_(y, *rbeQ)
    #     y = y + 1
    # end_timer('Qx_')
    #
    # return None, None

    # ipm main loop
    reset_timer('ipm')
    for it in range(ipm.n_iter):

        # 1. pre-conditioner
        reset_timer('pre-cond')
        invP = ipm_precond(ipm, s, z)
        end_timer('pre-cond')

        # 2. solve kkt 1
        reset_timer('kkt 1')
        dx_a, ds_a, dz_a = ipm_solve_rhs(ipm, s, z, invP, -r1, -r2, -r3, n_iter=ipm.n_pcg_iter_1)
        end_timer('kkt 1')

        # 3. centering parameters
        reset_timer('center')
        sigma, mu = centering_params(s, z, ds_a, dz_a, n_sample=ipm.n_linesearch)
        end_timer('center')

        # 4. solve kkt 2
        reset_timer('kkt 2')

        # option 1
        # r2 -= (sigma * mu - (ds_a * dz_a))
        # dx, ds, dz = ipm_solve_rhs(Q, s, z, invM, -r1, -r2, -r3, n_iter = pcg_iter, dx = dx_a, rbe = rbe)

        # option 2
        r2 = (sigma * mu - (ds_a * dz_a))
        dx, ds, dz = ipm_solve_rhs(ipm, s, z, invP, 0, r2, 0, n_iter=ipm.n_pcg_iter_2)
        dx, ds, dz = dx + dx_a, ds + ds_a, dz + dz_a
        end_timer('kkt 2')

        # 5. step size and update
        reset_timer('linesearch')
        alpha = 0.99 * linesearch(s, ds, z, dz, n_sample=ipm.n_linesearch)
        end_timer('linesearch')

        reset_timer('update')
        # update x
        x = x + alpha * dx
        s = s + alpha * ds
        z = z + alpha * dz

        # update result_x based on kkt residual
        r1, r2, r3, kkt_res = ipm_kkt_res(ipm, q, h, x, s, z)
        flag = kkt_res_best[inds] > kkt_res
        kkt_res_best[inds[flag]] = kkt_res[flag]
        result_x[:, inds[flag]] = x[:, flag]

        # remove converged
        flag = kkt_res > ipm.kkt_conv_eps
        inds = inds[flag]
        q, h, x, s, z, invP = q[:, flag], h[:, flag], x[:, flag], s[:, flag], z[:, flag], invP[:, flag]
        r1, r2, r3 = r1[:, flag], r2[:, flag], r3[:, flag]

        if inds.shape[0] == 0:
            break

        end_timer('update')

    xclip = torch.clip(result_x, xl, xu)
    velocity, velocity_inf_nrm = evaluate_result(xclip, ps, ipm)
    end_timer('ipm')
    return velocity.cpu().numpy(), (velocity_inf_nrm < rbe.velocity_tol).cpu().numpy()

def init_gurobi(parts, contacts, settings: dict):
    params = {
        "WLSACCESSID": "9d6cfee4-4a06-46b1-a7c8-a7445b4e62a6",
        "WLSSECRET" : "563345f3-3017-488e-a549-eb6742256f41",
        "LICENSEID": 2759892,
        "OptimalityTol": 1E-6,
        "OutputFlag": settings["rbe"]["verbose"],
        "Method": -1,
    }
    env = gp.Env(params = params)
    settings["gurobi"] = {
        "env": env,
        "pre-computed": True
    }


def simulate_gurobi(batch_part_states: list[dict],
                    settings: dict):
    env = settings["gurobi"]["env"]

    # rbe
    rbe = SimpleNamespace(**settings["rbe"])

    ps, cs = rbe.mapping(batch_part_states)
    q, xl, xu, Al, Au = compute_rbe_dynamic_attribs(rbe.g, ps, cs, rbe.Jn, rbe.Jt, rbe.invM, rbe.mu, rbe.Ccp)

    flags, vs = [], []
    nx, nA = q.shape[0], Al.shape[0]

    reset_timer('gurobi')
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
    vs = np.vstack(vs)
    vs = vs.T
    end_timer('gurobi')
    return vs, np.array(flags)

def simulate(parts: list[Trimesh],
             contacts: list[dict],
             batch_part_states: list[dict],
             settings: dict):
    rbe_pre_computed = settings.get("rbe", {"pre-computed": False}).get("pre-computed", False)
    if not rbe_pre_computed:
        init_rbe(parts, contacts, settings)

    if "gurobi" in settings:
        gurobi_pre_computed = settings["gurobi"].get("pre-computed", False)
        if not gurobi_pre_computed:
            init_gurobi(parts, contacts, settings)
        return simulate_gurobi(batch_part_states, settings)
    else:
        ipm_computed = settings.get("ipm", {"pre-computed": False}).get("pre-computed", False)
        if not ipm_computed:
            init_ipm(parts, contacts, settings)
        return simulate_ipm(batch_part_states, settings)

if __name__ == '__main__':
    from learn2assemble import ASSEMBLY_RESOURCE_DIR, default_settings, RESOURCE_DIR
    from learn2assemble.render import *
    from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
    import os

    # test
    default_settings['rbe']['density'] = 1000
    default_settings['rbe']['mu'] = 0.5
    default_settings['rbe']['Ccp'] = 5000
    default_settings["assembly"]["contact_shrink_ratio"] = 0  # for robustnessly computing the contact surfaces

    n_batch = 512
    torch.manual_seed(0)
    name = "dome"
    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + f"/{name}")

    filename = os.path.join(RESOURCE_DIR, f"curriculum/{name}.pt")
    part_states = torch.load(filename)['input']

    # choose the max parts
    inds = torch.sum(part_states, dim=1).cpu().numpy()
    inds = np.argsort(inds).tolist()[::-1]
    part_states = part_states[inds, :]

    # random
    # part_states = part_states[torch.randperm(part_states.shape[0]), :]
    part_states = part_states[:n_batch, :]

    default_settings['gurobi'] = {}
    default_settings['ipm'] = {
        "n_iter": 30,
        "n_pcg_iter_1": 200,
        "n_pcg_iter_2": 100,
        "n_pcg_eval_iter": 10,
        "n_linesearch": 32,
        "kkt_conv_eps": 1E-5,
        "x_bound_tol": 1E-6,
        "float_type": torch.float32,
    }
    contacts = compute_assembly_contacts(parts, default_settings)
    init_rbe(parts, contacts, default_settings)
    init_ipm(parts, contacts, default_settings)
    ipm = SimpleNamespace(**default_settings["ipm"])

    v_fp32, stable_fp32 = simulate(parts, contacts, part_states, default_settings)
    print_logger(n_batch)
    print(np.sum(stable_fp32) / n_batch)

    # render
    # import polyscope as ps
    # init_polyscope()
    # t = 0
    # def callback():
    #     global t
    #     changed, t = psim.SliderFloat("time", v=t, v_min=0, v_max=1)
    #     if changed:
    #         draw_assembly_motion(parts, part_states[0], v_fp32[:, 0] * t)
    #
    # draw_contacts(contacts, part_states[0])
    # draw_assembly_motion(parts, part_states[0], v_fp32[:, 0] * t)
    # ps.set_user_callback(callback)
    # ps.show()

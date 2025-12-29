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
def from_scipy_to_torch_sparse(A: sp.sparse.coo_matrix,
                               floatType=torch.float32):
    return torch.sparse_coo_tensor(torch.LongTensor(np.vstack((A.row, A.col))),
                                   torch.tensor(A.data, dtype=floatType),
                                   torch.Size(A.shape)).to(device)


def init_ipm(parts: list[Trimesh],
              contacts: list[dict],
              settings: dict):

    ipm = update_default_settings(settings,
                                   "ipm",
                                  {
                                       "ipm_iter": 25,
                                       "pcg_iter_1": 200,
                                       "pcg_iter_2": 100,
                                       "conv_eps": 1E-5,
                                       "pcg_eps": 1E-6,
                                       "x_eps": 1E-6,
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
    GT = G.transpose()
    GG = G * G

    ipm['diagQ'] = torch.tensor(Q.diagonal(), dtype=float_type, device=device)

    ipm['Q'] = torch.tensor(Q.todense(), dtype=float_type, device=device)
    ipm['G'] = torch.tensor(G.todense(), dtype=float_type, device=device)
    ipm['GT'] = torch.tensor(GT.todense(), dtype=float_type, device=device)
    ipm['GG'] = torch.tensor(GG.todense(), dtype=float_type, device=device)
    print(Q.nnz / Q.shape[0] / Q.shape[1])
    #ipm["Q"] = from_scipy_to_torch_sparse(Q.tocoo(), floatType=float_type)
    #ipm["G"] = from_scipy_to_torch_sparse(G, floatType=float_type)
    #ipm["GT"] = from_scipy_to_torch_sparse(GT, floatType=float_type)

    H = Q + G.T @ G
    H_tch = torch.tensor(H.todense(), dtype=torch.float64, device=device)
    cholesky_H = torch.linalg.cholesky(H_tch)
    ipm['invH'] = torch.cholesky_inverse(cholesky_H).type(float_type)
    ipm['pre-computed'] = True
    settings['ipm'] = ipm

def ipm_start_solve(invH, rbe, h, q, floatType):
    x = invH @ (GT_(h, rbe) - q)
    oldz = G_(x, rbe) - h
    alpha_p = torch.max(oldz, 0).values
    flag = (alpha_p < 0).type(floatType).repeat(oldz.shape[0], 1)
    s = flag * (-oldz) + (1 - flag) * (-oldz + (1 + alpha_p))

    alpha_d = -torch.min(oldz, 0).values
    flag = (alpha_d < 0).type(floatType).repeat(oldz.shape[0], 1)
    z = flag * (oldz) +  (1 - flag) * (oldz + 1 + alpha_d)
    return x, s, z

def GTZSG(p, ZS, rbe):
    λn, λt = p[:rbe.nλn, :], p[rbe.nλn: rbe.nλn + rbe.nλt, :]
    inds = torch.arange(rbe.nλn, device=p.device, dtype=torch.long)
    inds = inds.repeat(rbe.nt)
    λn = rbe.mu * λn
    λn.index_add_(0, inds, λt, alpha=-1.0)
    p = ZS * torch.vstack([-λn, -p, p])

    nx = rbe.nλn + rbe.nλt + rbe.nf
    λ0 = torch.zeros((nx, p.shape[1]), device=p.device, dtype=p.dtype)
    λ0[:rbe.nλn, :] = rbe.mu * (- p[:rbe.nλn])
    λ0[rbe.nλn : rbe.nλn + rbe.nλt, :] = p[:rbe.nλn].repeat((rbe.nt, 1))
    λ2 = p[rbe.nλn: rbe.nλn + nx]
    λ3 = p[rbe.nλn + nx:]
    return λ3 - λ2 + λ0

def GT_(p, rbe):
    nx = rbe.nλn + rbe.nλt + rbe.nf
    λ0 = torch.zeros((nx, p.shape[1]), device=p.device, dtype=p.dtype)
    λ0[:rbe.nλn, :] = rbe.mu * (- p[:rbe.nλn])
    λ0[rbe.nλn: rbe.nλn + rbe.nλt, :] = p[:rbe.nλn].repeat((rbe.nt, 1))
    λ2 = p[rbe.nλn: rbe.nλn + nx]
    λ3 = p[rbe.nλn + nx:]
    return λ3 - λ2 + λ0

def G_(p, rbe):
    λn, λt = p[:rbe.nλn, :], p[rbe.nλn: rbe.nλn + rbe.nλt, :]
    inds = torch.arange(rbe.nλn, device=p.device, dtype=torch.long)
    inds = inds.repeat(rbe.nt)
    λn = rbe.mu * λn
    λn.index_add_(0, inds, λt, alpha=-1.0)
    return torch.vstack([-λn, -p, p])

def ipm_kkt_res(Q, q, h, rbe, x, s, z):
    r1 = Q @ x + q + GT_(z, rbe)
    r2 = s * z
    r3 = G_(x, rbe) + s - h
    kkt_res = torch.vstack([r1, r2, r3])
    kkt_res = torch.abs(kkt_res)
    kkt_res = torch.max(kkt_res, 0).values
    return r1, r2, r3, kkt_res

def precond(diagQ, GG, s, z):
    ZS = z / s
    GTZSG = torch.einsum("ji, jb -> ib", GG, ZS)
    invM = GTZSG + diagQ[:, None]
    invM = 1.0 / invM
    return invM

def inf_norm(x):
    return torch.max(torch.abs(x), dim=0).values

def ipm_solve_rhs(Q, s, z, invM, v1, v2, v3, dx = None, eps = 1E-6, n_iter = 50, eval_it = 10, rbe = None):

    ZS = z / s
    b = GT_((z * v3 - v2) / s, rbe) + v1

    if dx is None:
        dx = torch.zeros_like(b)
        xk = torch.zeros_like(b)
        rk = b.clone()
    else:
        xk = dx.clone()
        rk = b - (GTZSG(dx, ZS, rbe) + Q @ dx)
        dx = torch.zeros_like(b)

    dx_rk = torch.ones(b.shape[1], device=device, dtype=b.dtype) * torch.inf

    uk = invM * rk
    pk = uk.clone()

    n_iter = n_iter // eval_it
    inds = torch.arange(b.shape[1], device=device, dtype=torch.long)
    #torch.cuda.synchronize()
    #timer = perf_counter()
    for k in range(n_iter):
        for t in range(eval_it):
            #Apk = GT @ (ZS * (G @ pk)) + Q @ pk
            Apk = GTZSG(pk, ZS, rbe) + Q @ pk
            ru = torch.sum(rk * uk, dim=0)
            ak = ru / torch.sum(pk * Apk, dim=0)
            xk += ak[None, :] * pk
            rk -= ak[None, :] * Apk
            uk = invM * rk
            betak = torch.sum(rk * uk, dim=0) / ru
            pk = uk + betak[None, :] * pk

        # to avoid numerical error
        error = inf_norm(rk)
        flag = error < dx_rk[inds]
        dx[:, inds[flag]] = xk[:, flag]
        dx_rk[inds[flag]] = error[flag]

        # flag = error > eps
        # inds = inds[flag]
        # xk, rk, uk, pk = xk[:, flag], rk[:, flag], uk[:, flag], pk[:, flag]
        # ZS, invM = ZS[:, flag], invM[:, flag]
        #
        # if inds.shape[0] == 0:
        #      break

        rk = b[:, inds] - (GTZSG(xk, ZS, rbe) + Q @ xk)
        uk = invM * rk

    #torch.cuda.synchronize()
    # pcg_time = (perf_counter() - timer)
    #print("pcg:\t", (perf_counter() - timer))

    ds = v3 - G_(dx, rbe)
    dz = (v2 - z * ds) / s

    return dx, ds, dz

def linesearch(s, ds, z, dz, nsample = 32):
    alpha = torch.linspace(0, 1, nsample, device=device, dtype=s.dtype)
    ls = s[None, :] + torch.einsum('i, jk -> ijk', alpha, ds)
    lz = z[None, :] + torch.einsum('i, jk -> ijk', alpha, dz)
    flag = torch.logical_and((ls >=0).all(dim = 1), (lz >=0).all(dim = 1))
    inds = torch.arange(nsample, device=device, dtype=s.dtype)
    inds = torch.einsum('ib, i -> ib', flag , inds)
    ind = torch.max(inds, dim = 0).values.to(torch.long)
    return alpha[ind]

def centering_params(s, z, ds_a, dz_a):
    """duality gap + cc term in predictor-corrector PDIP"""
    sz = torch.sum(s * z, dim = 0)
    mu =  sz / s.shape[0]
    alpha = linesearch(s, ds_a, z, dz_a)
    sigma = torch.sum((s + alpha * ds_a) * (z + alpha * dz_a), dim=0) / sz
    sigma = sigma ** 3
    return sigma, mu

def simulate_ipm(batch_part_states: list[dict],
                  settings: dict):

    # name space
    rbe = SimpleNamespace(**settings["rbe"])
    ipm = SimpleNamespace(**settings["ipm"])
    floatType = ipm.float_type

    ps, cs = rbe.mapping(batch_part_states)
    q, xl, xu, Al, Au = compute_rbe_dynamic_attribs(rbe.g, ps, cs, rbe.Jn, rbe.Jt, rbe.invM, rbe.mu, rbe.Ccp)

    xu += ipm.x_eps
    xl -= ipm.x_eps

    q = torch.tensor(q, dtype=floatType, device=device)
    xl = torch.tensor(xl, dtype=floatType, device=device)
    xu = torch.tensor(xu, dtype=floatType, device=device)
    Al = torch.tensor(Al, dtype=floatType, device=device)
    Au = torch.tensor(Au, dtype=floatType, device=device)
    h = torch.vstack([-Al, -xl, xu])

    # initialize ipm
    invH = ipm.invH
    GG = ipm.GG
    Q = ipm.Q
    diagQ = ipm.diagQ

    # start solve
    x, s, z = ipm_start_solve(invH, rbe, h, q, floatType)
    result_x = torch.zeros_like(x)
    inds = torch.arange(s.shape[1], dtype=torch.long, device=device)

    it = 0

    # check velocity
    g = torch.tensor(rbe.g, dtype=floatType, device=device)
    JnT = from_scipy_to_torch_sparse(rbe.Jn.transpose(), floatType=floatType)
    JtT = from_scipy_to_torch_sparse(rbe.Jt.transpose(), floatType=floatType)
    invMass = from_scipy_to_torch_sparse(rbe.invM, floatType=floatType)
    ps = torch.tensor(ps, dtype=floatType, device=device)


    # ipm
    torch.cuda.synchronize()
    timer_start = perf_counter()
    kkt_res_best = torch.ones(part_states.shape[0], device=device, dtype=floatType) * torch.inf

    r1, r2, r3, kkt_res = ipm_kkt_res(Q, q, h, rbe, x, s, z)
    while it < ipm.ipm_iter:

        #print("step ", it)
        #torch.cuda.synchronize()
        #timer = perf_counter()
        invM = precond(diagQ, GG, s, z)
        #torch.cuda.synchronize()
        #print("precond", perf_counter() - timer)

        #torch.cuda.synchronize()
        #timer = perf_counter()
        #torch.cuda.synchronize()
        #print("ipm_kkt_res", perf_counter() - timer)
        #print("kkt_res", torch.max(kkt_res, 0).values)
        #print("kkt_res", kkt_res)

        # update
        #torch.cuda.synchronize()
        #timer = perf_counter()
        dx_a, ds_a, dz_a = ipm_solve_rhs(Q, s, z, invM, -r1, -r2, -r3, n_iter = ipm.pcg_iter_1, eval_it=ipm.pcg_eval_it, rbe = rbe, eps = ipm.pcg_eps)
        #torch.cuda.synchronize()
        #print("ipm_solve_rhs_1", perf_counter() - timer)

        #torch.cuda.synchronize()
        #timer = perf_counter()
        sigma, mu = centering_params(s, z, ds_a, dz_a)

        #torch.cuda.synchronize()
        #print("centering_params", perf_counter() - timer)

        #torch.cuda.synchronize()
        #timer = perf_counter()
        # option 1
        # r2 -= (sigma * mu - (ds_a * dz_a))
        # dx, ds, dz = ipm_solve_rhs(Q, s, z, invM, -r1, -r2, -r3, eps = ipm.pcg_eps, n_iter = pcg_iter, dx = dx_a, rbe = rbe)

        # option 2
        r2 = (sigma * mu - (ds_a * dz_a))
        dx, ds, dz = ipm_solve_rhs(Q, s, z, invM, 0, r2, 0, n_iter= ipm.pcg_iter_2, eval_it=ipm.pcg_eval_it, rbe = rbe, eps = ipm.pcg_eps)
        dx, ds, dz = dx + dx_a, ds + ds_a, dz + dz_a

        #torch.cuda.synchronize()
        #print("ipm_solve_rhs_2", perf_counter() - timer)

        #torch.cuda.synchronize()
        #timer = perf_counter()
        alpha = 0.99 * linesearch(s, ds, z, dz)
        #torch.cuda.synchronize()
        #print("linesearch", perf_counter() - timer)

        x = x + alpha * dx
        s = s + alpha * ds
        z = z + alpha * dz

        # update inds
        r1, r2, r3, kkt_res = ipm_kkt_res(Q, q, h, rbe, x, s, z)
        flag = kkt_res_best[inds] > kkt_res
        kkt_res_best[inds[flag]] = kkt_res[flag]
        result_x[:, inds[flag]] = x[:, flag]

        # remove converged
        flag = kkt_res > ipm.conv_eps
        inds = inds[flag]
        q, h, x, s, z, invM = q[:, flag], h[:, flag], x[:, flag], s[:, flag], z[:, flag], invM[:, flag]
        r1, r2, r3 = r1[:, flag], r2[:, flag], r3[:, flag]

        if inds.shape[0] == 0:
            break

        #print(kkt_res)

        it = it + 1

    xclip = torch.clip(result_x, xl, xu)
    λn, λt = xclip[:rbe.nλn, :], xclip[rbe.nλn: rbe.nλn + rbe.nλt, :]
    residual = (JnT @ λn + JtT @ λt + g[:, None]) * ps
    velocity = invMass @ residual
    velocity_inf_nrm = torch.max(torch.abs(velocity), axis=0).values
    print("velocity", velocity_inf_nrm)
    print("avg time", (perf_counter() - timer_start)/part_states.shape[0])

    return velocity.cpu().numpy(), (velocity_inf_nrm < rbe.velocity_tol).cpu().numpy()


def init_admm(parts: list[Trimesh],
              contacts: list[dict],
              settings: dict):

    admm = update_default_settings(settings,
                       "admm", {
                           "sigma": 1E-6,
                           "r": 0.1,
                           "alpha": 1.6,
                           "max_iter": 2000,
                           "evaluate_iter": 100,
                           "float_type": torch.float32,
                       })

    rbe = settings['rbe']
    A = rbe["A"]
    L = rbe["L"]
    float_type = admm["float_type"]

    nx = A.shape[1]
    Inx = sp.sparse.coo_matrix(sp.sparse.eye_array(nx, dtype=np.float64))
    Ah = sp.sparse.block_array([[A],
                                [Inx]])
    AhT = Ah.transpose()

    admm["Ah"] = from_scipy_to_torch_sparse(Ah, floatType=float_type)
    admm["AhT"] = from_scipy_to_torch_sparse(AhT, floatType=float_type)
    admm["nx"] = nx
    admm["nA"] = A.shape[0]
    admm["nAh"] = Ah.shape[0]

    lhs = (L.T @ L + admm["sigma"] * Inx + admm["r"] * AhT @ Ah)
    lhs = torch.tensor(lhs.todense(), dtype=torch.float64, device=device)
    cholesky_lhs = torch.linalg.cholesky(lhs)
    admm["inv_lhs"] = torch.cholesky_inverse(cholesky_lhs).type(float_type)
    admm["pre-computed"] = True
    settings["admm"] = admm


def admm_step(xk, yk, zk, qk, zlk, zuk, Ah, AhT, inv_lhs, r, sigma, alpha, evaluate_iter):
    with torch.no_grad():
        for it in range(evaluate_iter):
            rhs = sigma * xk - qk + torch.sparse.mm(AhT, r * zk - yk)
            xh_k1 = (inv_lhs @ rhs)
            x_k1 = (xh_k1 * alpha + xk * (1 - alpha))
            zh_k1 = torch.sparse.mm(Ah, xh_k1)
            z_alpha = alpha * zh_k1 + (1 - alpha) * zk
            z_k1 = z_alpha + yk / r
            z_k1 = torch.clip(z_k1, zlk, zuk)
            y_k1 = yk + r * (z_alpha - z_k1)
            xk, yk, zk = x_k1.clone(), y_k1.clone(), z_k1.clone()
    return xk, yk, zk


def simulate_admm(batch_part_states: list[dict],
                  settings: dict):
    n_batch = batch_part_states.shape[0]

    # name space
    rbe = SimpleNamespace(**settings["rbe"])
    admm = SimpleNamespace(**settings["admm"])
    floatType = admm.float_type

    ps, cs = rbe.mapping(batch_part_states)
    q, xl, xu, Al, Au = compute_rbe_dynamic_attribs(rbe.g, ps, cs, rbe.Jn, rbe.Jt, rbe.invM, rbe.mu, rbe.Ccp)

    q = torch.tensor(q, dtype=floatType, device=device)
    xl = torch.tensor(xl, dtype=floatType, device=device)
    xu = torch.tensor(xu, dtype=floatType, device=device)
    Al = torch.tensor(Al, dtype=floatType, device=device)
    Au = torch.tensor(Au, dtype=floatType, device=device)
    zl = torch.vstack([Al, xl])
    zu = torch.vstack([Au, xu])

    x = torch.zeros(admm.nx, n_batch, device=device, dtype=floatType)
    y = torch.zeros(admm.nAh, n_batch, device=device, dtype=floatType)
    z = torch.zeros(admm.nAh, n_batch, device=device, dtype=floatType)
    batch_inds = torch.arange(n_batch, device=device, dtype=torch.long)
    stable_flags = torch.zeros(n_batch, device=device, dtype=torch.bool)

    g = torch.tensor(rbe.g, dtype=floatType, device=device)
    JnT = from_scipy_to_torch_sparse(rbe.Jn.transpose(), floatType=floatType)
    JtT = from_scipy_to_torch_sparse(rbe.Jt.transpose(), floatType=floatType)
    invM = from_scipy_to_torch_sparse(rbe.invM, floatType=floatType)
    ps = torch.tensor(ps, dtype=floatType, device=device)

    v = torch.zeros(rbe.nf, n_batch, device=device, dtype=floatType)

    torch.cuda.synchronize()
    timer_start = perf_counter()
    # admm iterations
    for iter in np.arange(stop=admm.max_iter, step=admm.evaluate_iter):

        xk, yk, zk = x[:, batch_inds].clone(), y[:, batch_inds].clone(), z[:, batch_inds].clone()
        qk, zlk, zuk = q[:, batch_inds].clone(), zl[:, batch_inds].clone(), zu[:, batch_inds].clone()

        # admm
        x[:, batch_inds], y[:, batch_inds], z[:, batch_inds] = admm_step(xk, yk, zk, qk, zlk, zuk,
                                                                         admm.Ah, admm.AhT, admm.inv_lhs, admm.r,
                                                                         admm.sigma, admm.alpha, admm.evaluate_iter)

        # evualuation
        xclip = torch.clip(xk, zlk[admm.nA:, :], zuk[admm.nA:, :])
        λn, λt = xclip[:rbe.nλn, :], xclip[rbe.nλn: rbe.nλn + rbe.nλt, :]
        residual = (torch.sparse.mm(JnT, λn) + torch.sparse.mm(JtT, λt) + g[:, None]) * ps[:, batch_inds]
        velocity = torch.sparse.mm(invM, residual)
        velocity_inf_nrm = torch.max(torch.abs(velocity), dim=0).values
        v[:, batch_inds] = velocity
        if rbe.verbose:
            print(velocity_inf_nrm)

        # termination
        stable_flags[batch_inds] = (velocity_inf_nrm < rbe.velocity_tol)
        not_terminated = ~stable_flags[batch_inds]
        batch_inds = batch_inds[not_terminated]
        if batch_inds.shape[0] == 0:
            break

    print("avg time", (perf_counter() - timer_start) / part_states.shape[0])

    return v.cpu().numpy(), stable_flags.cpu().numpy()

def init_gurobi(parts, contacts, settings: dict):
    env = gp.Env()
    env.setParam('OptimalityTol', 1E-6)
    env.setParam('OutputFlag', settings["rbe"]["verbose"])
    env.setParam('Method', -1)
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

    torch.cuda.synchronize()
    timer_start = perf_counter()
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
    print("avg time:\t", (perf_counter() - timer_start) / batch_part_states.shape[0])

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
    elif 'admm' in settings:
        # by default running admm
        admm_computed = settings.get("admm", {"pre-computed": False}).get("pre-computed", False)
        if not admm_computed:
            init_admm(parts, contacts, settings)
        return simulate_admm(batch_part_states, settings)
    elif 'ipm' in settings:
        ipm_computed = settings.get("ipm", {"pre-computed": False}).get("pre-computed", False)
        if not ipm_computed:
            init_ipm(parts, contacts, settings)
        return simulate_ipm(batch_part_states, settings)



if __name__ == '__main__':
    from learn2assemble import ASSEMBLY_RESOURCE_DIR, default_settings, RESOURCE_DIR
    from learn2assemble.render import *
    from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
    #import polyscope as ps
    import os

    #init_polyscope()

    # test
    default_settings['rbe']['density'] = 1000
    default_settings['rbe']['mu'] = 0.5
    default_settings['rbe']['Ccp'] = 500
    default_settings["assembly"]["contact_shrink_ratio"] = 0 # for robustnessly computing the contact surfaces

    n_batch = 512
    torch.manual_seed(0)
    name = "dome"
    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + f"/{name}")

    filename = os.path.join(RESOURCE_DIR, f"curriculum/{name}.pt")

    part_states = torch.load(filename)['input']
    inds = torch.sum(part_states, dim=1).cpu().numpy()
    inds = np.argsort(inds).tolist()[::-1]
    part_states = part_states[inds, :]

    #part_states = part_states[torch.randperm(part_states.shape[0]), :]
    part_states = part_states[:n_batch, :]

    # save data
    # dataset = {'input': part_states}
    # torch.save(dataset, os.path.join(RESOURCE_DIR, f"curriculum/{name}.pt"))
    # exit(0)

    default_settings.pop('admm', None)
    #default_settings['gurobi'] = {}
    default_settings['ipm'] = {
        'float_type': torch.float32,
        "ipm_iter": 30,
        "conv_eps": 1E-5,
        "pcg_iter_1": 200,
        "pcg_iter_2": 50,
        "pcg_eval_it": 10,
    }
    contacts = compute_assembly_contacts(parts, default_settings)
    init_rbe(parts, contacts, default_settings)
    init_ipm(parts, contacts, default_settings)

    v_fp32, stable_fp32 = simulate(parts, contacts, part_states, default_settings)
    #print(stable_fp32)
    print(np.sum(stable_fp32) / n_batch)

    # t = 0
    # def callback():
    #     global t
    #     changed, t = psim.SliderFloat("time", v=t, v_min=0, v_max=1)
    #     if changed:
    #         draw_assembly_motion(parts, part_states[0], v_fp32[:, 0] * t)

    # draw_contacts(contacts, part_states[0])
    # draw_assembly_motion(parts, part_states[0], v_fp32[:, 0] * t)
    # ps.set_user_callback(callback)
    # ps.show()

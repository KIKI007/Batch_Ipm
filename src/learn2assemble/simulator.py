import copy
from time import perf_counter
import gurobipy as gp
from gurobipy import GRB
from learn2assemble.rbe import *
from types import SimpleNamespace
import platform
from learn2assemble.rbe import num_vars

logger = {
    'timer' : {},
    'log': {},
    'activate': True,
}

def inf_norm(x):
    return torch.max(torch.abs(x), dim=0).values

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

def print_logger(nbatch = 1.0, names = []):
    if logger['activate']:
        if len(names) == 0:
            for name, value in logger['log'].items():
                print(name, f":\t\t\t {value / nbatch:.3e}")
        else:
            for name in names:
                if name in logger['log']:
                    print(name, f":\t\t\t {logger['log'][name] / nbatch:.3e}")

def GT_(p, nλn, nt, nf, mu):
    nλt = nλn * nt
    nx = nλn + nλt + nf
    λ0 = torch.zeros((nx, p.shape[1]), device=p.device, dtype=p.dtype)
    λ0[:nλn, :] = mu * (- p[:nλn])
    λ0[nλn: nλn + nλt, :] = p[:nλn].repeat((nt, 1))
    λ2 = p[nλn: nλn + nx]
    λ3 = p[nλn + nx:]
    return λ3 - λ2 + λ0

def G_(p, nλn, nt, nf, mu):
    nλt = nλn * nt
    λn, λt = p[:nλn, :], p[nλn: nλn + nλt, :]
    inds = torch.arange(nλn, device=p.device, dtype=torch.long)
    inds = inds.repeat(nt)
    λn = mu * λn
    λn.index_add_(0, inds, λt, alpha=-1.0)
    return torch.vstack([-λn, -p, p])

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


def init_ipm(parts: list[Trimesh],
             contacts: list[dict],
             settings: dict):

    ipm = update_default_settings(settings,
                                  "ipm",
                                  {
                                      "n_iter": 30,
                                      "n_pcg_iter": 100,
                                      "n_pcg_eval_iter": 20,
                                      "n_linesearch": 32,
                                      "kkt_conv_eps": 1E-5,
                                      "pcg_rel_eps": 1E-2,
                                      "x_bound_tol": 1E-6,
                                      "float_type": torch.float32,
                                      "compile": True,
                                      "device" : torch.device("cuda" if torch.cuda.is_available() else "cpu")})

    ipm = update_default_settings(settings,
                                  "ipm",
                                  settings["rbe"])
    device = ipm["device"]
    float_type = ipm['float_type']

    ipm['n_part'] = len(parts)
    ipm['boundary_part_ids'] = settings['env']['boundary_part_ids']
    ipm['density'] = compute_best_density(parts, ipm['boundary_part_ids'])
    ipm_contacts(ipm, parts, contacts, ipm['density'], ipm['boundary_part_ids'])

    if platform.system() == 'Windows' or not ipm['compile']:
        disable_compile = True
    else:
        disable_compile = False

    # compile function

    ipm['Q_'] = torch.compile(Q_, disable=disable_compile)
    ipm['GT_'] = torch.compile(GT_, disable=disable_compile)
    ipm['G_'] = torch.compile(G_, disable=disable_compile)
    ipm['GTZSG_'] = torch.compile(GTZSG_, disable=disable_compile)

    # compute pre-conditioner
    nx = ipm['nλn'] * (ipm["nt"] + 1) + ipm["nf"]
    rbeG = ipm["nλn"], ipm["nt"], ipm["nf"], ipm["mu"]
    rbeQ = ipm['nλn'], ipm['nt'], ipm['iAs'], ipm['iBs'], ipm['nAs'], ipm['nBs'], ipm['invM'], None
    p = torch.eye(nx, device=device, dtype=float_type)
    G = ipm['G_'](p, *rbeG)
    ipm['GG'] = G * G
    ipm['Q'] = ipm['Q_'](p, *rbeQ)
    ipm['diagQ'] = torch.diagonal(ipm['Q'])

    # cannot compile just use Q directly
    if disable_compile:
        ipm['Q_'] = lambda x, *args: args[-1] @ x
    else:
        torch.set_float32_matmul_precision('high')

    H = ipm['GT_'](G, *rbeG) + ipm['Q']
    cholesky_H = torch.linalg.cholesky(H)
    ipm['cholesky_H'] = cholesky_H

    ipm['pre-computed'] = True
    settings['ipm'] = ipm

def index_mapping(batch_part_states, iAs, iBs, nλn, device):
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

    print(device)
    ps, cs = index_mapping(batch_part_states, ipm.iAs, ipm.iBs, ipm.nλn, device = device)
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

def sum_forces(p, nλn, nt, iA, iB, nA, nB):
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
    residual = (sum_forces(xclip, *rbe) + ipm.g[:, None]) * ps
    residual = residual.reshape(-1, 3, batch)
    velocity = (ipm.invM @ residual).reshape(-1, batch)
    velocity_inf_nrm = inf_norm(velocity)
    return velocity, velocity_inf_nrm

def ipm_start_solve(ipm, h, q):
    rbe = ipm.nλn, ipm.nt, ipm.nf, ipm.mu
    b = ipm.GT_(h, *rbe) - q
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
    r1 = ipm.Q_(x, *rbeQ) + q + ipm.GT_(z, *rbe)
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

def centering_params(s, z, ds_a, dz_a, n_sample):
    """duality gap + cc term in predictor-corrector PDIP"""
    sz = torch.sum(s * z, dim=0)
    mu = sz / s.shape[0]
    alpha = ipm_linesearch(s, ds_a, z, dz_a, n_sample=n_sample)
    sigma = torch.sum((s + alpha * ds_a) * (z + alpha * dz_a), dim=0) / sz
    sigma = sigma ** 3
    return sigma, mu

def ipm_solve_rhs(ipm, s, z, invP, v1, v2, v3, n_iter, dx=None):
    device = ipm.device

    rbeG = ipm.nλn, ipm.nt, ipm.nf, ipm.mu
    rbeQ = ipm.nλn, ipm.nt, ipm.iAs, ipm.iBs, ipm.nAs, ipm.nBs, ipm.invM, ipm.Q

    ZS = z / s
    b = ipm.GT_((z * v3 - v2) / s, *rbeG) + v1

    if dx is None:
        dx = torch.zeros_like(b)
        xk = torch.zeros_like(b)
        rk = b.clone()
    else:
        xk = dx.clone()
        rk = b - (ipm.GTZSG_(dx, ZS, *rbeG) + ipm.Q_(dx, *rbeQ))
        dx = torch.zeros_like(b)

    dx_rk = torch.ones(b.shape[1], device=device, dtype=b.dtype) * 1E9

    uk = invP * rk
    pk = uk.clone()

    m = n_iter // ipm.n_pcg_eval_iter
    #reset_timer('pcg')
    for k in range(m):
        for t in range(ipm.n_pcg_eval_iter):
            # Apk = GT @ (ZS * (G @ pk)) + Q @ pk
            # fast computation
            Apk = ipm.GTZSG_(pk, ZS, *rbeG) + ipm.Q_(pk, *rbeQ)

            ru = torch.sum(rk * uk, dim=0)
            ak = ru / torch.sum(pk * Apk, dim=0)
            xk += ak[None, :] * pk
            rk -= ak[None, :] * Apk
            uk = invP * rk
            betak = torch.sum(rk * uk, dim=0) / ru
            pk = uk + betak[None, :] * pk

        # only update x when rk decrease
        prev_rk = dx_rk.clone()
        error = inf_norm(rk)
        flag = error < dx_rk
        dx[:, flag] = xk[:, flag]
        dx_rk[flag] = error[flag]
        rel = torch.max(torch.abs(dx_rk - prev_rk) / prev_rk)

        # recompute the residual to avoid numerical errors
        rk = b - (ipm.GTZSG_(xk, ZS, *rbeG) + ipm.Q_(xk, *rbeQ))
        uk = invP * rk

        if rel < ipm.pcg_rel_eps:
            break

    ds = v3 - ipm.G_(dx, *rbeG)
    dz = (v2 - z * ds) / s
    return dx, ds, dz

def get_dynamic_attrib(batch_part_states, rbe, float_type):
    ps, cs = rbe.mapping(batch_part_states)
    q, xl, xu, Al, Au = compute_rbe_dynamic_attribs(rbe.g, ps, cs, rbe.Jn, rbe.Jt, rbe.invM, rbe.mu, rbe.Ccp)
    q = torch.tensor(q, dtype=float_type, device=device)
    xl = torch.tensor(xl, dtype=float_type, device=device)
    xu = torch.tensor(xu, dtype=float_type, device=device)
    Al = torch.tensor(Al, dtype=float_type, device=device)
    Au = torch.tensor(Au, dtype=float_type, device=device)
    ps = torch.tensor(ps, dtype=float_type, device=device)
    cs = torch.tensor(cs, dtype=float_type, device=device)
    return q, xl, xu, Al, Au, ps, cs

def ipm_update_device(ipm, device):
    new_ipm = {}
    for name, val in ipm.items():
        if torch.is_tensor(val):
            new_ipm[name] = val.clone().to(device)
        else:
            new_ipm[name] = copy.deepcopy(val)
    new_ipm["device"] = device
    return new_ipm

def simulate_ipm(batch_part_states: list[dict], ipm_settings):

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
    inds = torch.arange(s.shape[1], dtype=torch.long, device=device)
    kkt_res_best = torch.ones(batch_part_states.shape[0], device=device, dtype=floatType) * torch.inf
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
        sigma, mu = centering_params(s, z, ds_a, dz_a, n_sample=ipm.n_linesearch)
        end_timer('center')

        # 4. solve kkt 2
        reset_timer('kkt 2')

        # option 1
        # r2 -= (sigma * mu - (ds_a * dz_a))
        # dx, ds, dz = ipm_solve_rhs(Q, s, z, invM, -r1, -r2, -r3, n_iter = pcg_iter, dx = dx_a, rbe = rbe)

        # option 2
        r2 = (sigma * mu - (ds_a * dz_a))
        dx, ds, dz = ipm_solve_rhs(ipm, s, z, invP, 0, r2, 0, n_iter=ipm.n_pcg_iter)
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
    velocity, velocity_inf_nrm = ipm_evaluate_result(ipm, xclip, ps)
    end_timer('ipm')
    return velocity, (velocity_inf_nrm < ipm.velocity_tol)

def ipm_auto_parameters(settings: dict):
    # decide Ccp
    settings["ipm"]["Ccp"] = 1.1 * abs(torch.sum(settings['ipm']['g']).item())
    settings["ipm"]["n_pcg_iter"] = int((settings["ipm"]["Q"].shape[0] * 0.02) // 10 * 10)
    print("density = ", settings["ipm"]["density"])
    print("Ccp = ", settings["ipm"]["Ccp"])
    print("num pcg iter = ", settings["ipm"]["n_pcg_iter"])
    return True

# for parallel gpus
class IpmSim(torch.nn.Module):
    def __init__(self, ipm_settings):
        super().__init__()
        self.ipm_settings = ipm_update_device(ipm_settings, device=ipm_settings['device'])
        for n, val in self.ipm_settings.items():
            if torch.is_tensor(val):
                self.register_buffer(n, tensor = self.ipm_settings[n], persistent=False)

    @torch.compile
    def Q_(self, p, nλn, nt, iA, iB, nA, nB, invM, Q):
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

    @torch.no_grad()
    def forward(self, x):
        new_settings = {}
        for n, val in self.ipm_settings.items():
            if torch.is_tensor(val):
                new_settings[n] = self.get_buffer(n)
                new_settings['device'] = new_settings[n].device
            else:
                new_settings[n] = copy.deepcopy(val)

        new_settings['Q_'] = self.Q

        velocity, stable_flag = simulate_ipm(x, new_settings)
        return stable_flag

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
    vs = np.vstack(vs)
    vs = vs.T
    end_timer('gurobi')
    return vs, np.array(flags)

def simulate(parts: list[Trimesh],
             contacts: list[dict],
             batch_part_states: list[dict],
             settings: dict):
    rbe_pre_computed = settings.get("rbe", {"pre-computed": False}).get("pre-computed", False)

    if "gurobi" in settings:
        if not rbe_pre_computed:
            init_rbe(parts, contacts, settings)
        gurobi_pre_computed = settings["gurobi"].get("pre-computed", False)
        if not gurobi_pre_computed:
            init_gurobi(parts, contacts, settings)
        return simulate_gurobi(batch_part_states, settings)
    else:
        ipm_computed = settings.get("ipm", {"pre-computed": False}).get("pre-computed", False)
        if not ipm_computed:
            init_ipm(parts, contacts, settings)
        return simulate_ipm(batch_part_states, settings["ipm"])

if __name__ == '__main__':
    from learn2assemble import ASSEMBLY_RESOURCE_DIR, default_settings, RESOURCE_DIR
    from learn2assemble.render import *
    from learn2assemble.assembly import load_assembly_from_files, compute_assembly_contacts
    import os

    # test

    default_settings['rbe']['mu'] = 0.5
    default_settings["assembly"]["contact_shrink_ratio"] = 0.0  # for robustnessly computing the contact surfaces

    n_batch = 512
    torch.manual_seed(0)
    name = "dome"
    parts = load_assembly_from_files(ASSEMBLY_RESOURCE_DIR + f"/{name}")
    default_settings['env']['boundary_part_ids'] = [len(parts) - 1]

    filename = os.path.join(RESOURCE_DIR, f"curriculum/{name}.pt")
    part_states = torch.load(filename)['input']

    # choose the max parts
    inds = torch.sum(part_states, dim=1).cpu().numpy()
    inds = np.argsort(inds).tolist()[::-1]
    part_states = part_states[inds, :]

    # random
    part_states = part_states[:n_batch, :]

    default_settings['ipm'] = {
        "n_iter": 30,
        "n_pcg_eval_iter": 10,
        "pcg_rel_eps": 0.1,
        "float_type": torch.float32,
        "compile": False,
    }
    reset_timer('contact')
    contacts = compute_assembly_contacts(parts, default_settings)
    end_timer('contact')

    reset_timer('init ipm')
    init_ipm(parts, contacts, default_settings)
    end_timer('init ipm')

    reset_timer('auto parameter')
    ipm_auto_parameters(settings = default_settings)
    end_timer('auto parameter')

    sim = IpmSim(default_settings["ipm"])
    parallel_sim = torch.nn.DataParallel(sim)

    torch.cuda.synchronize()
    timer = perf_counter()
    stable_fp32 = parallel_sim(part_states)

    torch.cuda.synchronize()
    print("time ", perf_counter() - timer)
    print(torch.sum(stable_fp32).item() / stable_fp32.shape[0])

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

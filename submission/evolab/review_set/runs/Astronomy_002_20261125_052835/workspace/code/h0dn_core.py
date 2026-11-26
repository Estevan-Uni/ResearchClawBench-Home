"""Core of the compact Local-Distance-Network GLS fit on the H0DN minimal dataset.

Model (magnitudes):
  primary row   : mu_obs(h, method, anchor)        = mu_h
  SN calibrator : m_B,h                              = mu_h + M_SN
  SBF calibrator: m_F110W,g                          = mu_Fornax + M_SBF      (sigma^2 += depth^2)
  group link    : 0                                  = mu_NGC1316 - mu_Fornax (sigma = depth)
  HF row        : m - 5log10(cz f(z)) - 25           = M_s - 5logH0           (sigma^2 += v_pec term)
  ext. intercept: 5 a_B + 25                         = 5logH0 - M_SN          (optional, literature a_B)

Anchor and method-anchor uncertainties are *not* parameters: they are folded into a
dense covariance C (C_ij += sigma_a^2 for rows sharing an anchor, += sigma_ma^2 for rows
sharing a method-anchor pair).  An explicit-nuisance version is provided for checking.
"""
import ast
import re
import numpy as np
from scipy import stats

C_KMS = 299792.458
LN10 = np.log(10.0)


# ----------------------------------------------------------------------------- parsing
def load_dataset(path):
    """Parse `name = <python literal>` blocks with ast.literal_eval (never exec)."""
    txt = open(path).read()
    lines = [l for l in txt.splitlines() if not l.strip().startswith('#')]
    txt = '\n'.join(lines)
    starts = [(m.start(), m.group(1)) for m in re.finditer(r'^([A-Za-z_]\w*)\s*=', txt, re.M)]
    out = {}
    for k, (pos, name) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(txt)
        rhs = txt[pos:end].split('=', 1)[1].strip()
        rhs = rhs.split('#')[0].strip() if '\n' not in rhs else rhs
        out[name] = ast.literal_eval(rhs)
    return out


# ----------------------------------------------------------------------------- helpers
def fz(z, q0=-0.55, j0=1.0):
    """R22 Eq. 4 expansion-history factor: d_L = cz f(z)/H0."""
    return 1 + 0.5 * (1 - q0) * z - (1 - q0 - 3 * q0 ** 2 + j0) * z ** 2 / 6.0


DEFAULTS = dict(
    drop_hosts=(),            # SN-calibrator hosts whose SN row is dropped
    drop_primary=(),          # (host, method, anchor) primary rows to drop
    use_fz=True, q0=-0.55, j0=1.0,
    sigv=None,                # override peculiar-velocity sigma (km/s); None = file value
    fornax_links=('NGC1316',),  # hosts tied to the Fornax group (TRGB route = NGC1316)
    hf_mode='rows',           # 'rows' | 'aggregate' | 'none'
    hf_samples=('SN', 'SBF'),
    zmin=0.0, zmax=1.0,
    hf_offset=0.0,            # magnitude offset added to *all* HF magnitudes (injection/what-if)
    hf_offset_sn=0.0, hf_offset_sbf=0.0,
    free_slope=False,         # free HF slope beta: y = M - 5logH0 + beta*(5log10 z - <5log10 z>)
    ext_aB=None,              # (a_B, sigma_aB) literature SN intercept row
    use_sbf=True,             # include SBF calibrators
    keep_virgo=False,         # keep NGC4472 (Virgo) with its own free mu_Virgo (self-determined)
    anchor_scale=1.0,
    extra_err=0.0,            # added in quadrature to every data row (Birge-like test)
    sn_sigint=0.0,            # SN Ia intrinsic scatter added to SN calibrator and HF SN rows
)


class System:
    """Holds L, y, C, and row/parameter bookkeeping for one network configuration."""

    def __init__(self, data, **opt):
        o = dict(DEFAULTS); o.update(opt); self.opt = o
        d = data
        anchors = {k: v for k, v in d['anchors'].items() if v['err'] > 0}  # drop MW (no data, zero err)
        self.anchors = anchors
        prim = [r for r in d['host_measurements'] if r[2] in anchors and tuple(r[:3]) not in o['drop_primary']]
        hosts = []
        for r in prim:
            if r[0] not in hosts:
                hosts.append(r[0])
        self.hosts = hosts
        params = ['mu_' + h for h in hosts]
        sn_rows = [r for r in d['sneia_calibrators'] if r[0] in hosts and r[0] not in o['drop_hosts']]
        groups = []
        if o['use_sbf'] and not any(h in hosts for h in o['fornax_links']):
            o['use_sbf'] = False      # no anchored path to Fornax -> SBF block carries no information
        if o['use_sbf']:
            groups.append('Fornax')
            if o['keep_virgo']:
                groups.append('Virgo')
        params += ['mu_' + g for g in groups]
        params += ['M_SN']
        use_sbf_hf = o['use_sbf'] and 'SBF' in o['hf_samples'] and o['hf_mode'] != 'none'
        if o['use_sbf']:
            params += ['M_SBF']
        self.has_H0 = (o['hf_mode'] != 'none') or (o['ext_aB'] is not None)
        if self.has_H0:
            params += ['5logH0']
        if o['free_slope']:
            params += ['beta_HF']
        self.params = params
        P = {p: i for i, p in enumerate(params)}
        rows, y, sig, info = [], [], [], []

        def add(coefs, val, s, meta):
            r = np.zeros(len(params))
            for k, c in coefs.items():
                r[P[k]] += c
            rows.append(r); y.append(val); sig.append(np.hypot(s, o['extra_err']) if meta['type'] != 'link' and meta['type'] != 'ext' else s); info.append(meta)

        for h, m, a, mu, e in prim:
            add({'mu_' + h: 1}, mu, e, dict(type='primary', host=h, method=m, anchor=a))
        for h, mb, e in sn_rows:
            add({'mu_' + h: 1, 'M_SN': 1}, mb, np.hypot(e, o['sn_sigint']), dict(type='sn_cal', host=h))
        if o['use_sbf']:
            for g_host, msbf, e in d['sbf_calibrators']:
                grp = d['host_group'].get(g_host)
                if grp not in groups:
                    continue
                add({'mu_' + grp: 1, 'M_SBF': 1}, msbf, np.hypot(e, d['depth_scatter']), dict(type='sbf_cal', host=g_host, group=grp))
            for h in o['fornax_links']:
                if h in hosts:
                    add({'mu_' + h: 1, 'mu_Fornax': -1}, 0.0, d['depth_scatter'], dict(type='link', host=h, group='Fornax'))
        # Hubble flow
        hf = []
        if o['hf_mode'] != 'none':
            if 'SN' in o['hf_samples']:
                hf += [('SN',) + tuple(r) for r in d['hubble_flow_sneia']]
            if use_sbf_hf:
                hf += [('SBF',) + tuple(r) for r in d['hubble_flow_sbf']]
        hf = [r for r in hf if o['zmin'] <= r[1] < o['zmax']]
        lz_ref = np.mean([5 * np.log10(r[1]) for r in hf]) if hf else 0.0
        agg = {}
        for s, z, m, e, v in hf:
            v = o['sigv'] if o['sigv'] is not None else v
            m = m + o['hf_offset'] + (o['hf_offset_sn'] if s == 'SN' else o['hf_offset_sbf'])
            f = fz(z, o['q0'], o['j0']) if o['use_fz'] else 1.0
            yy = m - 5 * np.log10(C_KMS * z * f) - 25
            sv = 5 * v / (LN10 * C_KMS * z)
            ss = np.sqrt(e ** 2 + sv ** 2 + (o['sn_sigint'] ** 2 if s == 'SN' else 0.0))
            Mk = 'M_SN' if s == 'SN' else 'M_SBF'
            if o['hf_mode'] == 'rows':
                coefs = {Mk: 1, '5logH0': -1}
                if o['free_slope']:
                    coefs['beta_HF'] = 5 * np.log10(z) - lz_ref
                add(coefs, yy, ss, dict(type='hf_' + s.lower(), z=z, m=m, sig_m=e, sig_v=sv))
            else:
                agg.setdefault(s, []).append((yy, ss))
        for s, lst in agg.items():  # one aggregated intercept row per sample (R22 Eq. 4-5)
            yy = np.array([a for a, b in lst]); w = 1 / np.array([b for a, b in lst]) ** 2
            Mk = 'M_SN' if s == 'SN' else 'M_SBF'
            add({Mk: 1, '5logH0': -1}, np.sum(w * yy) / w.sum(), 1 / np.sqrt(w.sum()), dict(type='hf_agg_' + s.lower(), n=len(lst)))
        if o['ext_aB'] is not None:
            aB, saB = o['ext_aB']
            add({'5logH0': 1, 'M_SN': -1}, 5 * aB + 25, 5 * saB, dict(type='ext', source='R22 a_B'))
        self.L = np.array(rows); self.y = np.array(y); self.sig = np.array(sig); self.info = info
        self.n, self.p = self.L.shape
        # covariance: diagonal + shared anchor + shared method-anchor
        C = np.diag(self.sig ** 2)
        self.C_blocks = {'stat_' + t: np.zeros_like(C) for t in dict.fromkeys(i['type'] for i in info)}
        for k, i in enumerate(info):
            self.C_blocks['stat_' + i['type']][k, k] = self.sig[k] ** 2
        self.C_blocks['anchor'] = np.zeros_like(C)
        self.C_blocks['method_anchor'] = np.zeros_like(C)
        mae = d['method_anchor_err']
        for i, a in enumerate(info):
            for j, b in enumerate(info):
                if a['type'] == 'primary' and b['type'] == 'primary':
                    if a['anchor'] == b['anchor']:
                        v = (anchors[a['anchor']]['err'] * o['anchor_scale']) ** 2
                        C[i, j] += v; self.C_blocks['anchor'][i, j] += v
                        if a['method'] == b['method']:
                            v = mae[(a['method'], a['anchor'])] ** 2
                            C[i, j] += v; self.C_blocks['method_anchor'][i, j] += v
        self.C = C

    def types(self):
        return np.array([i['type'] for i in self.info])


def gls(L, y, C):
    Ci = np.linalg.inv(C)
    F = L.T @ Ci @ L
    rank = np.linalg.matrix_rank(F)
    cov = np.linalg.inv(F)
    A = cov @ L.T @ Ci
    q = A @ y
    r = y - L @ q
    chi2 = float(r @ Ci @ r)
    return dict(q=q, cov=cov, A=A, r=r, chi2=chi2, rank=rank, Ci=Ci)


def fit(data, **opt):
    S = System(data, **opt)
    R = gls(S.L, S.y, S.C)
    R['sys'] = S
    R['dof'] = S.n - S.p
    R['P'] = {p: i for i, p in enumerate(S.params)}
    if S.has_H0:
        k = R['P']['5logH0']
        R['H0'] = 10 ** (R['q'][k] / 5)
        R['sH0'] = R['H0'] * LN10 / 5 * np.sqrt(R['cov'][k, k])
    else:
        R['H0'] = R['sH0'] = np.nan
    R['pval'] = float(stats.chi2.sf(R['chi2'], R['dof'])) if R['dof'] > 0 else np.nan
    return R


def par(R, name):
    k = R['P'][name]
    return R['q'][k], np.sqrt(R['cov'][k, k])


def clip_fit(data, thresh=3.3, types=('primary', 'sn_cal', 'sbf_cal'), **opt):
    """R22-style iterative global sigma clipping (one-by-one, largest |r|/sqrt(Var(r)) first).
    Clipped SN rows are removed via drop_hosts, primary rows via drop_primary."""
    opt = dict(opt)
    removed = []
    while True:
        R = fit(data, **opt)
        S = R['sys']
        Vr = S.C - S.L @ R['cov'] @ S.L.T  # covariance of residuals
        z = R['r'] / np.sqrt(np.clip(np.diag(Vr), 1e-12, None))
        cand = [k for k, i in enumerate(S.info) if i['type'] in types]
        if not cand:
            break
        k = max(cand, key=lambda k: abs(z[k]))
        if abs(z[k]) < thresh:
            break
        i = S.info[k]
        removed.append((i['type'], i.get('host'), i.get('method'), i.get('anchor'), float(z[k])))
        if i['type'] == 'sn_cal':
            opt['drop_hosts'] = tuple(opt.get('drop_hosts', ())) + (i['host'],)
        elif i['type'] == 'primary':
            opt['drop_primary'] = tuple(opt.get('drop_primary', ())) + ((i['host'], i['method'], i['anchor']),)
        else:
            break
    R['clipped'] = removed
    R['clip_opt'] = opt
    return R


def explicit_nuisance(data, **opt):
    """Same network with anchor offsets and method-anchor offsets as explicit parameters
    and prior rows (0 = dA +- sigma_A, 0 = dMA +- sigma_MA); C becomes diagonal."""
    S = System(data, **opt)
    o = S.opt
    prim_idx = [k for k, i in enumerate(S.info) if i['type'] == 'primary']
    anc = sorted(set(S.info[k]['anchor'] for k in prim_idx))
    ma = sorted(set((S.info[k]['method'], S.info[k]['anchor']) for k in prim_idx))
    extra = ['dA_' + a for a in anc] + ['dMA_%s_%s' % m for m in ma]
    L = np.hstack([S.L, np.zeros((S.n, len(extra)))])
    for k in prim_idx:
        i = S.info[k]
        L[k, S.p + anc.index(i['anchor'])] = 1
        L[k, S.p + len(anc) + ma.index((i['method'], i['anchor']))] = 1
    prior_L = np.zeros((len(extra), L.shape[1])); prior_s = []
    for j, e in enumerate(extra):
        prior_L[j, S.p + j] = 1
        if e.startswith('dA_'):
            prior_s.append(S.anchors[e[3:]]['err'] * o['anchor_scale'])
        else:
            m, a = e[4:].split('_')
            prior_s.append(data['method_anchor_err'][(m, a)])
    L = np.vstack([L, prior_L]); y = np.concatenate([S.y, np.zeros(len(extra))])
    C = np.diag(np.concatenate([S.sig, prior_s]) ** 2)
    R = gls(L, y, C)
    R['params'] = S.params + extra
    R['n'], R['p'] = L.shape
    R['dof'] = R['n'] - R['p']
    k = S.params.index('5logH0')
    R['H0'] = 10 ** (R['q'][k] / 5)
    R['sH0'] = R['H0'] * LN10 / 5 * np.sqrt(R['cov'][k, k])
    return R

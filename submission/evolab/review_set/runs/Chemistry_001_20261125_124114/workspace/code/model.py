"""Compact AlphaFold3-style network: input embedder -> Pairformer-lite trunk -> atom-level
EDM diffusion module (atom encoder / token transformer / atom decoder) + distogram head.
Dimensions are reduced so that training runs on 2 CPU cores."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from biomol import RESTYPES, MOLTYPES, ELEMENTS

N_RES, N_MOL, N_ELEM = len(RESTYPES), len(MOLTYPES), len(ELEMENTS)


def LN(d):
    return nn.LayerNorm(d)


class Transition(nn.Module):
    def __init__(self, c, n=2):
        super().__init__()
        self.ln = LN(c); self.a = nn.Linear(c, n * c, bias=False); self.b = nn.Linear(c, n * c, bias=False)
        self.o = nn.Linear(n * c, c, bias=False)

    def forward(self, x):
        x = self.ln(x)
        return self.o(F.silu(self.a(x)) * self.b(x))


class TriMul(nn.Module):
    """Triangle multiplicative update (outgoing / incoming)."""
    def __init__(self, c, outgoing=True):
        super().__init__()
        self.out = outgoing
        self.ln = LN(c); self.ab = nn.Linear(c, 4 * c, bias=False); self.g = nn.Linear(c, c, bias=False)
        self.lo = LN(c); self.o = nn.Linear(c, c, bias=False)

    def forward(self, z):
        z = self.ln(z)
        a, ga, b, gb = self.ab(z).chunk(4, -1)
        a = a * torch.sigmoid(ga); b = b * torch.sigmoid(gb)
        if self.out:
            x = torch.einsum('ikc,jkc->ijc', a, b)
        else:
            x = torch.einsum('kic,kjc->ijc', a, b)
        return torch.sigmoid(self.g(z)) * self.o(self.lo(x))


class AttnPairBias(nn.Module):
    """Gated multi-head self-attention with pair bias; optional AdaLN conditioning."""
    def __init__(self, c, c_z, heads=4, c_cond=None):
        super().__init__()
        self.h = heads; self.d = c // heads
        self.cond = c_cond is not None
        if self.cond:
            self.ada = AdaLN(c, c_cond)
            self.gate_out = nn.Linear(c_cond, c)
            nn.init.constant_(self.gate_out.bias, -2.0)
        else:
            self.ln = LN(c)
        self.qkvg = nn.Linear(c, 4 * c, bias=False)
        self.zb = nn.Sequential(LN(c_z), nn.Linear(c_z, heads, bias=False))
        self.o = nn.Linear(c, c, bias=False)

    def forward(self, a, z, s=None, mask_bias=None):
        x = self.ada(a, s) if self.cond else self.ln(a)
        q, k, v, g = self.qkvg(x).chunk(4, -1)
        sh = x.shape[:-1]
        q, k, v = [t.reshape(*sh, self.h, self.d) for t in (q, k, v)]
        logits = torch.einsum('...ihd,...jhd->...hij', q, k) / math.sqrt(self.d)
        logits = logits + self.zb(z).movedim(-1, -3)
        if mask_bias is not None:
            logits = logits + mask_bias
        w = logits.softmax(-1)
        o = torch.einsum('...hij,...jhd->...ihd', w, v).reshape(*sh, -1)
        o = self.o(torch.sigmoid(g) * o)
        if self.cond:
            o = torch.sigmoid(self.gate_out(s)) * o
        return o


class AdaLN(nn.Module):
    def __init__(self, c, c_s):
        super().__init__()
        self.ln = nn.LayerNorm(c, elementwise_affine=False)
        self.ls = nn.LayerNorm(c_s, bias=False)
        self.sc = nn.Linear(c_s, c); self.sh = nn.Linear(c_s, c, bias=False)

    def forward(self, a, s):
        s = self.ls(s)
        return torch.sigmoid(self.sc(s)) * self.ln(a) + self.sh(s)


class CondTransition(nn.Module):
    def __init__(self, c, c_s, n=2):
        super().__init__()
        self.ada = AdaLN(c, c_s)
        self.a = nn.Linear(c, n * c, bias=False); self.b = nn.Linear(c, n * c, bias=False)
        self.o = nn.Linear(n * c, c, bias=False); self.g = nn.Linear(c_s, c)
        nn.init.constant_(self.g.bias, -2.0)

    def forward(self, a, s):
        x = self.ada(a, s)
        return torch.sigmoid(self.g(s)) * self.o(F.silu(self.a(x)) * self.b(x))


class DiffTransformer(nn.Module):
    def __init__(self, c, c_s, c_z, blocks, heads):
        super().__init__()
        self.att = nn.ModuleList([AttnPairBias(c, c_z, heads, c_s) for _ in range(blocks)])
        self.tr = nn.ModuleList([CondTransition(c, c_s) for _ in range(blocks)])

    def forward(self, a, s, z, mask_bias=None):
        for at, tr in zip(self.att, self.tr):
            b = at(a, z, s, mask_bias)
            a = b + tr(a, s)  # AF3 parallel residual form
        return a


class PairformerBlock(nn.Module):
    def __init__(self, c_s, c_z, heads=4):
        super().__init__()
        self.tmo = TriMul(c_z, True); self.tmi = TriMul(c_z, False)
        self.zt = Transition(c_z, 2)
        self.att = AttnPairBias(c_s, c_z, heads)
        self.st = Transition(c_s, 2)

    def forward(self, s, z):
        z = z + self.tmo(z)
        z = z + self.tmi(z)
        z = z + self.zt(z)
        s = s + self.att(s, z)
        s = s + self.st(s)
        return s, z


def relpos(f, r_max=32, s_max=2):
    """AF3 relative position encoding (Algorithm 3)."""
    asym, res, tokidx, ent, sym = f['asym_id'], f['residue_index'], f['token_index'], f['entity_id'], f['sym_id']
    same_chain = asym[:, None] == asym[None]
    same_res = (res[:, None] == res[None]) & same_chain
    same_ent = ent[:, None] == ent[None]
    d_res = torch.where(same_chain, torch.clamp(res[:, None] - res[None] + r_max, 0, 2 * r_max),
                        torch.full_like(same_chain, 2 * r_max + 1, dtype=torch.long))
    d_tok = torch.where(same_res, torch.clamp(tokidx[:, None] - tokidx[None] + r_max, 0, 2 * r_max),
                        torch.full_like(same_chain, 2 * r_max + 1, dtype=torch.long))
    d_ch = torch.where(~same_chain, torch.clamp(sym[:, None] - sym[None] + s_max, 0, 2 * s_max),
                       torch.full_like(same_chain, 2 * s_max + 1, dtype=torch.long))
    return torch.cat([F.one_hot(d_res, 2 * r_max + 2), F.one_hot(d_tok, 2 * r_max + 2),
                      same_ent[..., None].float(), F.one_hot(d_ch, 2 * s_max + 2)], -1).float()


class AtomFeat(nn.Module):
    """Per-atom single features and local atom-pair features from the reference conformer."""
    def __init__(self, c_atom, c_ap, use_ref=True):
        super().__init__()
        self.use_ref = use_ref
        d_in = 3 + 1 + 1 + N_ELEM + 4 * 64 + N_MOL
        self.emb = nn.Linear(d_in, c_atom, bias=False)
        self.pd = nn.Linear(3, c_ap, bias=False); self.pinv = nn.Linear(1, c_ap, bias=False)
        self.pv = nn.Linear(1, c_ap, bias=False)
        self.cl = nn.Linear(c_atom, c_ap, bias=False); self.cm = nn.Linear(c_atom, c_ap, bias=False)
        self.mlp = nn.Sequential(nn.ReLU(), nn.Linear(c_ap, c_ap, bias=False), nn.ReLU(),
                                 nn.Linear(c_ap, c_ap, bias=False), nn.ReLU(), nn.Linear(c_ap, c_ap, bias=False))

    def forward(self, f, qi, ki, kmask):
        rp = f['ref_pos'] * (1.0 if self.use_ref else 0.0)
        rm = f['ref_mask'][:, None] * (1.0 if self.use_ref else 0.0)
        x = torch.cat([rp, rm, torch.arcsinh(f['ref_charge'])[:, None],
                       F.one_hot(f['ref_element'], N_ELEM).float(),
                       F.one_hot(f['ref_atom_name_chars'], 64).float().flatten(1),
                       F.one_hot(f['atom_moltype'], N_MOL).float()], -1)
        c = self.emb(x)
        # local pairs: queries qi (B,32) keys ki (B,128)
        d = rp[qi][:, :, None] - rp[ki][:, None]
        uid = f['ref_space_uid']
        v = ((uid[qi][:, :, None] == uid[ki][:, None]) & kmask[:, None]).float()[..., None]
        if not self.use_ref:
            v = v * 0
        p = self.pd(d) * v + self.pinv(1.0 / (1.0 + (d ** 2).sum(-1, keepdim=True))) * v + self.pv(v) * v
        p = p + self.cl(F.relu(c))[qi][:, :, None] + self.cm(F.relu(c))[ki][:, None]
        p = p + self.mlp(p)
        return c, p


class AtomTransformer(nn.Module):
    """Sequence-local atom attention: blocks of 32 query atoms x 128 key atoms (AF3)."""
    def __init__(self, c, c_ap, blocks=2, heads=4):
        super().__init__()
        self.att = nn.ModuleList([LocalAttn(c, c_ap, heads) for _ in range(blocks)])
        self.tr = nn.ModuleList([CondTransition(c, c) for _ in range(blocks)])

    def forward(self, q, c, p, qi, ki, kmask):
        for at, tr in zip(self.att, self.tr):
            b = at(q, c, p, qi, ki, kmask)
            q = b + tr(q, c)
        return q


class LocalAttn(nn.Module):
    def __init__(self, c, c_ap, heads):
        super().__init__()
        self.h = heads; self.d = c // heads
        self.ada = AdaLN(c, c)
        self.q = nn.Linear(c, c); self.kvg = nn.Linear(c, 3 * c, bias=False)
        self.pb = nn.Sequential(LN(c_ap), nn.Linear(c_ap, heads, bias=False))
        self.o = nn.Linear(c, c, bias=False); self.go = nn.Linear(c, c)
        nn.init.constant_(self.go.bias, -2.0)

    def forward(self, q, c, p, qi, ki, kmask):
        # q: (S, N, c) [S = diffusion batch], c: (N, c) or (S,N,c)
        x = self.ada(q, c)
        N = q.shape[-2]
        Q = self.q(x); K, V, G = self.kvg(x).chunk(3, -1)
        S = q.shape[0]
        Qb = Q[:, qi].reshape(S, *qi.shape, self.h, self.d)          # S,B,32,h,d
        Kb = K[:, ki].reshape(S, *ki.shape, self.h, self.d)          # S,B,128,h,d
        Vb = V[:, ki].reshape(S, *ki.shape, self.h, self.d)
        lg = torch.einsum('sbqhd,sbkhd->sbhqk', Qb, Kb) / math.sqrt(self.d)
        lg = lg + self.pb(p).permute(0, 3, 1, 2)[None]
        lg = lg.masked_fill(~kmask[None, :, None, None, :], -1e9)
        o = torch.einsum('sbhqk,sbkhd->sbqhd', lg.softmax(-1), Vb).reshape(S, -1, self.h * self.d)
        o = o[:, :N]  # qi blocks cover 0..Npad-1 contiguously; drop padding
        o = self.o(torch.sigmoid(G) * o)
        return torch.sigmoid(self.go(c)) * o


def local_blocks(N, nq=32, nk=128):
    nb = (N + nq - 1) // nq
    qi = torch.arange(nb * nq).reshape(nb, nq).clamp(max=N - 1)
    starts = torch.arange(nb) * nq + nq // 2 - nk // 2
    ki_raw = starts[:, None] + torch.arange(nk)[None]
    kmask = (ki_raw >= 0) & (ki_raw < N)
    return qi, ki_raw.clamp(0, N - 1), kmask


class FourierEmb(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.register_buffer('w', torch.randn(c)); self.register_buffer('b', torch.rand(c))

    def forward(self, t):
        return torch.cos(2 * math.pi * (t[..., None] * self.w + self.b))


class AF3Lite(nn.Module):
    def __init__(self, c_s=128, c_z=32, c_atom=64, c_ap=16, c_tok=128, n_pairformer=4,
                 n_diff_blocks=4, n_atom_blocks=2, heads=4, sigma_data=16.0,
                 use_trunk=True, use_bonds=True, use_ref=True, n_dist_bins=64, canon=False, canon_mode='pca'):
        super().__init__()
        self.sd = sigma_data
        self.use_trunk, self.use_bonds = use_trunk, use_bonds
        self.canon = canon
        self.canon_mode = canon_mode
        self.c_s, self.c_z = c_s, c_z
        # input embedder
        self.atomfeat_in = AtomFeat(c_atom, c_ap, use_ref)
        self.atom_enc_in = AtomTransformer(c_atom, c_ap, 1, heads)
        self.agg_in = nn.Linear(c_atom, c_s, bias=False)
        c_in = c_s + N_RES + N_MOL
        self.s_init = nn.Linear(c_in, c_s, bias=False)
        self.zi = nn.Linear(c_in, c_z, bias=False); self.zj = nn.Linear(c_in, c_z, bias=False)
        self.rp = nn.Linear(2 * 66 + 1 + 6, c_z, bias=False)
        self.bond = nn.Linear(1, c_z, bias=False)
        self.pairformer = nn.ModuleList([PairformerBlock(c_s, c_z, heads) for _ in range(n_pairformer)])
        self.disto = nn.Linear(c_z, n_dist_bins)
        self.n_bins = n_dist_bins
        # diffusion module
        self.dc_z = nn.Sequential(LN(c_z + 139), nn.Linear(c_z + 139, c_z, bias=False))
        self.dc_zt = nn.ModuleList([Transition(c_z, 2) for _ in range(2)])
        self.dc_s = nn.Sequential(LN(c_s + c_in), nn.Linear(c_s + c_in, c_s, bias=False))
        self.fourier = FourierEmb(256)
        self.dc_n = nn.Sequential(LN(256), nn.Linear(256, c_s, bias=False))
        self.dc_st = nn.ModuleList([Transition(c_s, 2) for _ in range(2)])
        self.atomfeat = AtomFeat(c_atom, c_ap, use_ref)
        self.enc_s = nn.Sequential(LN(c_s), nn.Linear(c_s, c_atom, bias=False))
        self.enc_z = nn.Sequential(LN(c_z), nn.Linear(c_z, c_ap, bias=False))
        self.enc_r = nn.Linear(3, c_atom, bias=False)
        self.atom_enc = AtomTransformer(c_atom, c_ap, n_atom_blocks, heads)
        self.agg = nn.Linear(c_atom, c_tok, bias=False)
        self.tok_s = nn.Sequential(LN(c_s), nn.Linear(c_s, c_tok, bias=False))
        self.tok_tf = DiffTransformer(c_tok, c_s, c_z, n_diff_blocks, heads)
        self.tok_ln = LN(c_tok)
        self.dec_in = nn.Linear(c_tok, c_atom, bias=False)
        self.atom_dec = AtomTransformer(c_atom, c_ap, n_atom_blocks, heads)
        self.dec_out = nn.Sequential(LN(c_atom), nn.Linear(c_atom, 3, bias=False))
        nn.init.zeros_(self.dec_out[1].weight)

    # ---------------------------------------------------------------- helpers
    def _blocks(self, f):
        N = f['ref_pos'].shape[0]
        if '_blocks' not in f:
            f['_blocks'] = local_blocks(N)
        return f['_blocks']

    @staticmethod
    def agg_atoms(x, a2t, T):
        # mean over atoms of each token; x (..., N, c)
        out = torch.zeros(*x.shape[:-2], T, x.shape[-1], dtype=x.dtype)
        out = out.index_add(-2, a2t, x)
        cnt = torch.bincount(a2t, minlength=T).clamp(min=1).to(x.dtype)
        return out / cnt[:, None]

    # ---------------------------------------------------------------- trunk
    def trunk(self, f):
        qi, ki, km = self._blocks(f)
        T = f['restype'].shape[0]
        a2t = f['atom_to_token']
        c, p = self.atomfeat_in(f, qi, ki, km)
        q = self.atom_enc_in(c[None], c, p, qi, ki, km)[0]
        a = self.agg_atoms(F.relu(self.agg_in(q)), a2t, T)
        s_inputs = torch.cat([a, F.one_hot(f['restype'], N_RES).float(), F.one_hot(f['moltype'], N_MOL).float()], -1)
        s = self.s_init(s_inputs)
        rel = relpos(f)
        z = self.zi(s_inputs)[:, None] + self.zj(s_inputs)[None] + self.rp(rel)
        if self.use_bonds:
            z = z + self.bond(f['token_bonds'][..., None])
        if self.use_trunk:
            for blk in self.pairformer:
                s, z = blk(s, z)
        disto_logits = self.disto(z + z.transpose(0, 1))
        return dict(s_inputs=s_inputs, s=s, z=z, rel=rel, disto=disto_logits)

    # ---------------------------------------------------------------- diffusion
    def conditioning(self, tr, sigma):
        z = self.dc_z(torch.cat([tr['z'], tr['rel']], -1))
        for t in self.dc_zt:
            z = z + t(z)
        s = self.dc_s(torch.cat([tr['s'], tr['s_inputs']], -1))
        n = self.fourier(0.25 * torch.log(sigma / self.sd))  # (S,256)
        s = s[None] + self.dc_n(n)[:, None]
        for t in self.dc_st:
            s = s + t(s)
        return s, z

    @staticmethod
    def canonical_frame(x):
        """Principal-axis frame of each point cloud x (S,N,3): returns centre (S,1,3) and rotation
        V (S,3,3) whose columns are the axes (descending variance), signs fixed by the third moment
        and the last axis chosen to make V right-handed (preserves chirality). No gradient."""
        with torch.no_grad():
            mu = x.mean(1, keepdim=True)
            xc = x - mu
            C = torch.einsum('sni,snj->sij', xc, xc) / x.shape[1]
            evals, evecs = torch.linalg.eigh(C.double())
            V = evecs.flip(-1).float()                       # descending variance
            proj = torch.einsum('sni,sij->snj', xc, V)
            sgn = torch.sign((proj ** 3).sum(1))
            sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
            V = V * sgn[:, None, :]
            det = torch.det(V)
            V = torch.cat([V[:, :, :2], V[:, :, 2:] * det[:, None, None]], -1)
        return mu, V

    @staticmethod
    def sequence_frame(x, f):
        """Sequence-anchored equivariant frame: u1 = sum_i (t_i - mean t) x_i, u2 = sum_i cos(2 pi t_i) x_i,
        t_i = normalised token position of atom i; Gram-Schmidt, e3 = e1 x e2 (right-handed)."""
        with torch.no_grad():
            T = f['restype'].shape[0]
            t = (f['atom_to_token'].float() + 0.5) / T
            mu = x.mean(1, keepdim=True)
            xc = x - mu
            u1 = torch.einsum('n,snj->sj', t - t.mean(), xc)
            u2 = torch.einsum('n,snj->sj', torch.cos(2 * math.pi * t), xc)
            e1 = u1 / u1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            u2 = u2 - (u2 * e1).sum(-1, keepdim=True) * e1
            e2 = u2 / u2.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            V = torch.stack([e1, e2, torch.cross(e1, e2, dim=-1)], -1)
        return mu, V

    def denoise(self, f, tr, x_noisy, sigma):
        """x_noisy (S,N,3), sigma (S,). Returns denoised positions (S,N,3).
        With canon=True the network operates in the canonical frame of x_noisy (SE(3)-equivariant
        by construction); otherwise it relies on random-rotation augmentation (AF3)."""
        if self.canon:
            mu, V = self.canonical_frame(x_noisy) if self.canon_mode == 'pca' else self.sequence_frame(x_noisy, f)
            xr = torch.einsum('sni,sij->snj', x_noisy - mu, V)
            out = self._denoise(f, tr, xr, sigma)
            return torch.einsum('snj,sij->sni', out, V) + mu
        return self._denoise(f, tr, x_noisy, sigma)

    def _denoise(self, f, tr, x_noisy, sigma):
        qi, ki, km = self._blocks(f)
        a2t = f['atom_to_token']
        T = f['restype'].shape[0]
        s, z = self.conditioning(tr, sigma)
        r = x_noisy / torch.sqrt(sigma ** 2 + self.sd ** 2)[:, None, None]
        c, p = self.atomfeat(f, qi, ki, km)
        cS = c[None] + self.enc_s(s)[:, a2t]                       # S,N,c_atom (trunk single broadcast)
        tq, tk = a2t[qi], a2t[ki]
        p = p + self.enc_z(z)[tq[:, :, None], tk[:, None]]
        q = cS + self.enc_r(r)
        q = self.atom_enc(q, cS, p, qi, ki, km)
        a = self.agg_atoms(F.relu(self.agg(q)), a2t, T)
        a = a + self.tok_s(s)
        a = self.tok_tf(a, s, z)
        a = self.tok_ln(a)
        q = self.dec_in(a)[:, a2t] + q
        q = self.atom_dec(q, cS, p, qi, ki, km)
        r_up = self.dec_out(q)
        sg = sigma[:, None, None]
        return self.sd ** 2 / (self.sd ** 2 + sg ** 2) * x_noisy + sg * self.sd / torch.sqrt(self.sd ** 2 + sg ** 2) * r_up

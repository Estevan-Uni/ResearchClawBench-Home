"""Fast sparse simulator for flyvis connectome-constrained DMNs.

Implements exactly the PPNeuronIGRSynapses dynamics of flyvis:
    tau_i dV_i/dt = -V_i + b_i + sum_j w_ij relu(V_j) + x_i(t)
    w_ij = sign_(ti,tj) * N_(ti,tj,du,dv) * alpha_(ti,tj)
forward Euler with step dt, effective tau = max(tau, dt), initial state = bias,
followed by `t_pre` seconds of grey (0.5) input -> steady state (as in flyvis).

The weight matrix is stored as a scipy CSR matrix (targets x sources); batched
simulation therefore needs one sparse mat-mat product per time step.

The module exposes per-type / per-edge-type parameter tables so that ablations
(edge masks, parameter overrides, coarse-graining, random initialisations) can
be built without touching flyvis internals.
"""
import os
os.environ.setdefault("FLYVIS_ROOT_DIR", "/workspace/outputs/flyvis_root")
import logging
import numpy as np
import scipy.sparse as sp
import torch
torch.set_num_threads(2)

logging.getLogger("flyvis").setLevel(logging.WARNING)

ENSEMBLE = "flow/0000"
MODEL_NAMES = [f"{ENSEMBLE}/{i:03d}" for i in range(50)]
DATA_DIR = "/workspace/data/flow/0000"

_NET = None


def get_network():
    """Return a single (re-usable) flyvis Network instance (connectome is identical
    for all models, only parameters differ)."""
    global _NET
    if _NET is None:
        from flyvis import NetworkView
        nv = NetworkView(MODEL_NAMES[0])
        _NET = nv.init_network()
        _NET.eval()
        for p in _NET.parameters():
            p.requires_grad = False
    return _NET


def load_state(i_or_name, net=None, which="best_chkpt"):
    net = net or get_network()
    name = i_or_name if isinstance(i_or_name, str) else f"{i_or_name:03d}"
    path = f"{DATA_DIR}/{name}/{which}" if which == "best_chkpt" else f"{DATA_DIR}/{name}/chkpts/{which}"
    chk = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(chk["network"])
    net.clamp()
    return chk


class Topology:
    """Static connectome information (shared by all models)."""

    def __init__(self, net=None):
        net = net or get_network()
        c = net.connectome
        self.n_nodes = int(len(c.nodes.type[:]))
        self.node_type = c.nodes.type[:].astype(str)
        self.u = c.nodes.u[:].astype(int)
        self.v = c.nodes.v[:].astype(int)
        self.src = net._source_indices.numpy().astype(np.int64)
        self.tgt = net._target_indices.numpy().astype(np.int64)
        self.n_edges = len(self.src)
        e = c.edges
        self.edge_src_type = e.source_type[:].astype(str)
        self.edge_tgt_type = e.target_type[:].astype(str)
        self.edge_du = e.du[:].astype(int)
        self.edge_dv = e.dv[:].astype(int)
        self.edge_nsyn = e.n_syn[:].astype(float)
        self.edge_sign_raw = e.sign[:].astype(float)
        self.cell_types = list(dict.fromkeys(self.node_type))  # ordered unique
        self.input_index = np.asarray(net.stimulus.input_index)  # (8, n_hex)
        self.central_cells_index = np.asarray(c.central_cells_index[:])
        self.layer_index = {ct: np.asarray(idx) for ct, idx in c.nodes.layer_index.items()}
        # readers for parameters
        self.readers = {}
        for name, p in {**net.node_params, **net.edge_params}.items():
            r = p.readers
            key = "nodes" if "nodes" in r else "edges"
            self.readers[name] = np.asarray(r[key])
        self.param_keys = {name: list(p.keys) for name, p in {**net.node_params, **net.edge_params}.items()}
        # u, v of input hexals
        r1 = self.layer_index["R1"]
        self.u_in, self.v_in = self.u[r1], self.v[r1]


_TOPO = None


def topo():
    global _TOPO
    if _TOPO is None:
        _TOPO = Topology()
    return _TOPO


def semantic_params(net=None):
    """Return dict of semantic (per-type / per-edge-type) parameter vectors of the
    currently loaded network (after clamp)."""
    net = net or get_network()
    out = {}
    for name, p in {**net.node_params, **net.edge_params}.items():
        out[name] = p.semantic_values.detach().cpu().numpy().astype(np.float64).copy()
    return out


def model_params(i, which="best_chkpt"):
    load_state(i, which=which)
    return semantic_params()


class FastSim:
    """Sparse forward-Euler simulator built from semantic parameters (torch CSR,
    state layout (N, B))."""

    def __init__(self, params, edge_mask=None, w_override=None, custom_edges=None, dtype=torch.float32):
        T = topo()
        self.T = T
        bias = params["bias"][T.readers["bias"]]
        tau = params["time_const"][T.readers["time_const"]]
        sign = params["sign"][T.readers["sign"]]
        cnt = params["syn_count"][T.readers["syn_count"]]
        strength = params["syn_strength"][T.readers["syn_strength"]]
        w = sign * cnt * strength if w_override is None else w_override.copy()
        self.tgt_e, self.src_e = T.tgt, T.src
        if custom_edges is not None:
            self.tgt_e, self.src_e, w = [np.asarray(a) for a in custom_edges]
        if edge_mask is not None:
            w = w * edge_mask
        self.w_edges = w
        self.dtype = dtype
        self.bias = torch.tensor(bias, dtype=dtype)[:, None]
        self.tau = torch.tensor(tau, dtype=dtype)[:, None]
        coo = sp.coo_matrix((w, (self.tgt_e, self.src_e)), shape=(T.n_nodes, T.n_nodes)).tocsr()
        coo.sum_duplicates()
        self.W = torch.sparse_csr_tensor(torch.tensor(coo.indptr, dtype=torch.int64),
                                         torch.tensor(coo.indices, dtype=torch.int64),
                                         torch.tensor(coo.data, dtype=dtype),
                                         size=(T.n_nodes, T.n_nodes))
        # photoreceptor input map: hex index -> node indices (8 per hexal)
        self.in_nodes = torch.tensor(T.input_index.reshape(-1), dtype=torch.int64)
        self.n_rep = T.input_index.shape[0]

    def _x(self, frame_hex):
        """frame_hex: torch (n_hex, B) -> (len(in_nodes), B) external current."""
        return frame_hex.repeat(self.n_rep, 1)

    def _step(self, V, xin, dt):
        R = torch.relu(V)
        drive = self.bias + torch.sparse.mm(self.W, R)
        drive[self.in_nodes] += xin
        tau = torch.clamp(self.tau, min=dt)
        return V + dt / tau * (drive - V)

    @torch.no_grad()
    def steady_state(self, dt, t_pre=1.0, value=0.5):
        V = self.bias.clone()
        n = int(t_pre / dt)
        xin = self._x(torch.full((self.T.input_index.shape[1], 1), value, dtype=self.dtype))
        for _ in range(n):
            V = self._step(V, xin, dt)
        return V  # (N,1)

    @torch.no_grad()
    def simulate(self, movie, dt, t_pre=1.0, record="central", V0=None, return_last=False,
                 currents_onto=None, every=1):
        """movie: (B, T, n_hex) numpy/torch luminance. Starts from steady state after
        t_pre s of grey (computed with batch 1, as flyvis). record: 'central' ->
        (B,T,n_types) voltages of central cells, 'all' -> (B,T,N), or index array.
        currents_onto: node index -> also returns synaptic currents w*relu(V_src)
        of all edges onto that node (B,T,n_in), computed from the state before each
        Euler step (i.e. the current that drives the next state)."""
        movie = torch.as_tensor(np.asarray(movie, dtype=np.float32))
        B, nT, _ = movie.shape
        if V0 is None:
            V0 = self.steady_state(dt, t_pre=t_pre)
        V0 = torch.as_tensor(V0, dtype=self.dtype)
        V = V0.expand(-1, B).clone() if V0.shape[1] == 1 else V0.clone()
        if isinstance(record, str) and record == "central":
            idx = torch.tensor(self.T.central_cells_index, dtype=torch.int64)
        elif isinstance(record, str) and record == "all":
            idx = None
        else:
            idx = torch.as_tensor(record, dtype=torch.int64)
        out, cur = [], []
        if currents_onto is not None:
            in_edges = np.where(self.tgt_e == currents_onto)[0]
            in_src = torch.tensor(self.src_e[in_edges], dtype=torch.int64)
            in_w = torch.tensor(self.w_edges[in_edges], dtype=self.dtype)[:, None]
        mv = movie.permute(1, 2, 0).contiguous()  # (T, hex, B)
        for t in range(nT):
            if currents_onto is not None:
                cur.append((torch.relu(V[in_src]) * in_w).T.clone())
            V = self._step(V, self._x(mv[t]), dt)
            if t % every == 0:
                out.append((V if idx is None else V[idx]).T.clone())
        out = torch.stack(out, 1).numpy()
        res = [out]
        if currents_onto is not None:
            res += [torch.stack(cur, 1).numpy(), in_edges]
        if return_last:
            res.append(V)
        return res[0] if len(res) == 1 else tuple(res)


def central_type_order():
    T = topo()
    return [T.node_type[i] for i in T.central_cells_index]

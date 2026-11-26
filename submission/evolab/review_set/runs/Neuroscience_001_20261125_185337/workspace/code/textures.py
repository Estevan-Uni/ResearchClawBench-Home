"""Synthetic optic-flow stimuli (Sintel is not downloadable here): a 1/f-noise texture
moved rigidly (translation in 8 directions, rotation, expansion), rendered to the hex
lattice with flyvis' BoxEye(extent=15, kernel_size=13) at dt = 1/50 s (training dt),
plus the decoder of each trained model to read out optic flow."""
import sys
sys.path.insert(0, "/workspace/code")
import numpy as np
import torch
import torch.nn.functional as F
import fastsim as fs
from flyvis.datasets.rendering import BoxEye

EYE = BoxEye(extent=15, kernel_size=13)
H, W = [int(x) for x in EYE.min_frame_size.tolist()]
# receptor centres (y,x) relative to frame centre in pixels
RC = EYE.receptor_centers.cpu().numpy().astype(float)  # (721,2) y,x offsets


def pink_noise(n=1024, seed=0, beta=1.6):
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n)[:, None] ** 2 + np.fft.fftfreq(n)[None] ** 2
    f[0, 0] = 1
    spec = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / f ** (beta / 2)
    img = np.real(np.fft.ifft2(spec))
    img = (img - img.mean()) / img.std()
    return np.clip(0.5 + 0.18 * img, 0, 1).astype(np.float32)


def _warp(img, frames_affine):
    """Sample frames (T,H,W) from img via affine maps given as (T,2,3) in pixel units
    (output pixel (x,y) relative to frame centre -> texture coordinates)."""
    n = img.shape[0]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    xs -= W / 2; ys -= H / 2
    out = []
    timg = torch.tensor(img)[None, None]
    for A in frames_affine:
        tx = A[0, 0] * xs + A[0, 1] * ys + A[0, 2]
        ty = A[1, 0] * xs + A[1, 1] * ys + A[1, 2]
        grid = torch.tensor(np.stack([tx / (n / 2), ty / (n / 2)], -1))[None].float()
        out.append(F.grid_sample(timg, grid, align_corners=False, mode="bilinear", padding_mode="reflection")[0, 0])
    return torch.stack(out)


def translation_movies(img, angles_deg=range(0, 360, 45), speed_px=4.0, n_frames=40):
    """Texture translation: image content moves with velocity speed_px pixels/frame in
    direction angle (0 = rightward, 90 = upward on screen). Returns hex movies
    (B,T,721) and the ground-truth flow direction (B,2) in (x right, y up)."""
    movies, gts = [], []
    for a in angles_deg:
        vx, vy = speed_px * np.cos(np.deg2rad(a)), speed_px * np.sin(np.deg2rad(a))
        A = []
        for t in range(n_frames):
            # content moving by +v means sample texture at x - v t ; screen y axis points down
            A.append(np.array([[1, 0, -vx * t], [0, 1, +vy * t]], dtype=np.float32))
        fr = _warp(img, A)
        movies.append(EYE(fr[None]).reshape(n_frames, -1).numpy())
        gts.append([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
    return np.stack(movies), np.array(gts)


def rotation_expansion_movies(img, n_frames=40, omega=0.012, k=0.012):
    """Rotation (counter-clockwise on screen) and expansion around the centre.
    Returns movies (2,T,721) and gt flow (2,2,721) in x-right/y-up at receptors."""
    movies = []
    for mode in ["rotation", "expansion"]:
        A = []
        for t in range(n_frames):
            if mode == "rotation":
                th = omega * t  # sampling at R(+th)x (y-down) rotates content ccw in y-up coords
                c, s = np.cos(th), np.sin(th)
                A.append(np.array([[c, -s, 0], [s, c, 0]], dtype=np.float32))
            else:
                sc = 1 / (1 + k * t)
                A.append(np.array([[sc, 0, 0], [0, sc, 0]], dtype=np.float32))
        fr = _warp(img, A)
        movies.append(EYE(fr[None]).reshape(n_frames, -1).numpy())
    # gt flow at receptor positions (x right, y up)
    y = -RC[:, 0]; x = RC[:, 1]
    rot = np.stack([-y, x]) * omega
    exp = np.stack([x, y]) * k
    return np.stack(movies), np.stack([rot, exp])


_DEC = {}


def decoder(i):
    if i not in _DEC:
        from flyvis import NetworkView
        nv = NetworkView(fs.MODEL_NAMES[i])
        d = nv.init_decoder()["flow"]
        d.eval()
        _DEC[i] = d
    return _DEC[i]


@torch.no_grad()
def decode(sim, movies, model_i, dt=1 / 50, t_static=0.5, n_last=20):
    """Simulate full network on movies (B,T,721) after grey steady-state + t_static s of
    the first frame, decode flow with model_i's decoder. Returns (B, n_last, 2, 721)."""
    B, nT, _ = movies.shape
    Vg = sim.steady_state(dt, 1.0)
    first = np.repeat(movies[:, :1], int(t_static / dt), 1)
    _, V = sim.simulate(first, dt=dt, V0=Vg, record=np.array([0]), return_last=True)
    act = sim.simulate(movies, dt=dt, V0=V, record="all")  # (B,T,N)
    dec = decoder(model_i)
    out = dec(torch.tensor(act[:, -n_last:]))
    return out.numpy()


def flow_scores(dec_flow, gt_dir):
    """dec_flow (B,T,2,721), gt_dir (B,2): cosine of spatially+temporally averaged decoded
    flow with gt direction, mean per-hexal cosine, angular error (deg) of mean vector."""
    m = dec_flow.mean(axis=(1, 3))  # (B,2)
    cos = (m * gt_dir).sum(1) / (np.linalg.norm(m, axis=1) + 1e-12)
    per = dec_flow.mean(1)  # (B,2,721)
    pc = (per * gt_dir[:, :, None]).sum(1) / (np.linalg.norm(per, axis=1) + 1e-12)
    ang = np.rad2deg(np.abs(np.angle(np.exp(1j * (np.arctan2(m[:, 1], m[:, 0]) - np.arctan2(gt_dir[:, 1], gt_dir[:, 0]))))))
    return dict(cos=cos, hex_cos=pc.mean(1), ang_err=ang, gain=np.linalg.norm(m, axis=1))

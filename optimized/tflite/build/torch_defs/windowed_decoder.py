"""Windowed SAME-L decoder: a drop-in DifferentialSWA.forward replacement (O(S) blocked ±17 attn,
identical params so weights load unchanged). Importing this patches the class. Verified == dense."""
import sys, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))   # same_l_decoder_torch lives beside this file
import same_l_decoder_torch as M
BLK = M.BLOCK_SIZE  # 17


def _win_mask(nb, device, dtype):
    S = BLK * nb
    b = torch.arange(nb, device=device)
    gq = b[:, None] * BLK + torch.arange(BLK, device=device)[None, :]              # [nb,17]
    gk = b[:, None] * BLK - BLK + torch.arange(3 * BLK, device=device)[None, :]    # [nb,51]
    band = (gq[:, :, None] - gk[:, None, :]).abs() <= BLK
    valid = (gk[:, None, :] >= 0) & (gk[:, None, :] < S)
    ok = band & valid
    return torch.where(ok, torch.zeros((), device=device, dtype=dtype),
                       torch.full((), float("-inf"), device=device, dtype=dtype))


def _windowed_attn(Q, K, V, scale, S):
    B, TwoH, _, hd = Q.shape
    nb = S // BLK
    Qb = Q.reshape(B, TwoH, nb, BLK, hd); Kb = K.reshape(B, TwoH, nb, BLK, hd); Vb = V.reshape(B, TwoH, nb, BLK, hd)
    z = torch.zeros(B, TwoH, 1, BLK, hd, device=Q.device, dtype=Q.dtype)
    Kwin = torch.cat([torch.cat([z, Kb[:, :, :-1]], 2), Kb, torch.cat([Kb[:, :, 1:], z], 2)], dim=3)
    Vwin = torch.cat([torch.cat([z, Vb[:, :, :-1]], 2), Vb, torch.cat([Vb[:, :, 1:], z], 2)], dim=3)
    scores = torch.matmul(Qb, Kwin.transpose(-1, -2)) * scale + _win_mask(nb, Q.device, Q.dtype)
    return torch.matmul(torch.softmax(scores, dim=-1), Vwin).reshape(B, TwoH, S, hd)


def _windowed_forward(self, x, cos, sin, attn_mask=None):
    B, T, _ = x.shape
    H, D = M.NUM_HEADS, M.HEAD_DIM
    q1, k1, v, q2, k2 = self.to_qkv(x).chunk(5, dim=-1)
    th = lambda t: t.view(B, T, H, D).transpose(1, 2)
    q1, k1, v, q2, k2 = [th(t) for t in (q1, k1, v, q2, k2)]
    q1, k1, q2, k2 = self.q_norm(q1), self.k_norm(k1), self.q_norm(q2), self.k_norm(k2)
    q1, k1 = M._apply_rope(q1, cos, sin), M._apply_rope(k1, cos, sin)
    q2, k2 = M._apply_rope(q2, cos, sin), M._apply_rope(k2, cos, sin)
    Q = torch.cat([q1, q2], 1); K = torch.cat([k1, k2], 1); V = torch.cat([v, v], 1)
    out = _windowed_attn(Q, K, V, self.scale, T)
    out1, out2 = out.chunk(2, dim=1)
    return self.to_out((out1 - out2).transpose(1, 2).reshape(B, T, M.DIM))


def patch():
    M.DifferentialSWA.forward = _windowed_forward


if __name__ == "__main__":
    # full-decoder equivalence: dense vs windowed, real weights, output_audio
    dense = M.load_model(T_lat=None, output_audio=True).eval()
    outs = {}
    for T_lat in (16, 64, 128):
        lat = torch.randn(1, 256, T_lat)
        with torch.no_grad():
            outs[T_lat] = (lat, dense(lat))
    patch()
    win = M.load_model(T_lat=None, output_audio=True).eval()
    for T_lat, (lat, yd) in outs.items():
        with torch.no_grad():
            yw = win(lat)
        d = (yw - yd).abs().max().item()
        rms = 20 * torch.log10((yw - yd).pow(2).mean().sqrt() / (yd.pow(2).mean().sqrt() + 1e-9) + 1e-12).item()
        print(f"T_lat={T_lat:<4d} full-decoder windowed vs dense: max|d|={d:.2e} rms={rms:.1f}dB  out{tuple(yw.shape)}")

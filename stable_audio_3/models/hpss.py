import torch
import torch.nn.functional as F

def _enhance_either_hpss(x_padded, out, kernel_size, power, which, offset):
    """x_padded: one that median filtering can be directly applied
    kernel_size = int
    dim: either 2 (freq-axis) or 3 (time-axis)
    which: str, either "harm" or "perc"
    """
    if which == "harm":
        for t in range(out.shape[3]):
            out[:, :, :, t] = torch.median(x_padded[:, :, offset:-offset, t:t + kernel_size], dim=3)[0]

    elif which == "perc":
        for f in range(out.shape[2]):
            out[:, :, f, :] = torch.median(x_padded[:, :, f:f + kernel_size, offset:-offset], dim=2)[0]
    else:
        raise NotImplementedError("it should be either but you passed which={}".format(which))

    if power != 1.0:
        out.pow_(power)


def hpss(x, kernel_size=31, power=2.0, hard=False):
    """x: |STFT| (or any 2-d representation) in batch, (not in a decibel scale!)
    in a shape of (batch, ch, freq, time)
    power: to which the enhanced spectrograms are used in computing soft masks.
    kernel_size: odd-numbered {int or tuple of int}
        if tuple,
            1st: width of percussive-enhancing filter (one along freq axis)
            2nd: width of harmonic-enhancing filter (one along time axis)
        if int,
            it's applied for both perc/harm filters
    """
    eps = 1e-7
    if isinstance(kernel_size, tuple):
        pass
    else:
        # pad is int
        kernel_size = (kernel_size, kernel_size)

    pad = (kernel_size[0] // 2, kernel_size[0] // 2,
           kernel_size[1] // 2, kernel_size[1] // 2,)

    harm, perc, ret = torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)
    x_padded = F.pad(x, pad=pad, mode='reflect')

    _enhance_either_hpss(x_padded, out=perc, kernel_size=kernel_size[0], power=power, which='perc', offset=kernel_size[1]//2)
    _enhance_either_hpss(x_padded, out=harm, kernel_size=kernel_size[1], power=power, which='harm', offset=kernel_size[0]//2)

    if hard:
        mask_harm = harm > perc
        mask_perc = harm < perc
    else:
        mask_harm = (harm + eps) / (harm + perc + eps)
        mask_perc = (perc + eps) / (harm + perc + eps)

    return x * mask_harm, x * mask_perc, mask_harm, mask_perc
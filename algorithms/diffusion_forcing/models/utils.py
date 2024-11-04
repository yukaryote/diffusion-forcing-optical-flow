import math
import torch
from torch import nn
from einops import rearrange, parse_shape


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    f, b = t.shape
    out = a[t]
    return out.reshape(f, b, *((1,) * (len(x_shape) - 2)))


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# old sigmoid_beta_schedule
# def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
#     """
#     sigmoid schedule
#     proposed in https://arxiv.org/abs/2212.11972 - Figure 8
#     better for images > 64x64, when used during training
#     """
#     steps = timesteps + 1
#     t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
#     v_start = torch.tensor(start / tau).sigmoid()
#     v_end = torch.tensor(end / tau).sigmoid()
#     alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
#     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#     return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas_cumprod = enforce_zero_terminal_snr(alphas_cumprod[1:])
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = torch.cat([alphas_cumprod[0:1], alphas])
    betas = 1 - alphas
    return torch.clip(betas, 1e-9, 0.99999)

def enforce_zero_terminal_snr(alphas_cumprod):
    """
    enforce zero terminal SNR following https://arxiv.org/abs/2305.08891
    returns betas
    """
    alphas_cumprod_sqrt = torch.sqrt(alphas_cumprod)

    # store old values
    alphas_cumprod_sqrt_0 = alphas_cumprod_sqrt[0].clone()
    alphas_cumprod_sqrt_T = alphas_cumprod_sqrt[-1].clone()
    # shift so last timestep is zero
    alphas_cumprod_sqrt -= alphas_cumprod_sqrt_T
    # scale so first timestep is back to original value
    alphas_cumprod_sqrt *= alphas_cumprod_sqrt_0 / alphas_cumprod_sqrt[0]
    # convert to betas
    alphas_cumprod = alphas_cumprod_sqrt**2
    assert alphas_cumprod[-1] == 0, "terminal SNR not zero"
    return alphas_cumprod


class EinopsWrapper(nn.Module):
    def __init__(self, from_shape: str, to_shape: str, module: nn.Module):
        super().__init__()
        self.module = module
        self.from_shape = from_shape
        self.to_shape = to_shape

    def forward(self, x: torch.Tensor, *args, **kwargs):
        axes_lengths = parse_shape(x, pattern=self.from_shape)
        x = rearrange(x, f"{self.from_shape} -> {self.to_shape}")
        x = self.module(x, *args, **kwargs)
        try:
            x = rearrange(x, f"{self.to_shape} -> {self.from_shape}", **axes_lengths)
        except:
            #HACK: ignore axes lengths when we concat condition to input
            # or else the input and output axes lengths will not match
            # and will throw an error
            x = rearrange(x, f"{self.to_shape} -> {self.from_shape}")
        return x


def get_einops_wrapped_module(module, from_shape: str, to_shape: str):
    class WrappedModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.wrapper = EinopsWrapper(from_shape, to_shape, module(*args, **kwargs))

        def forward(self, x: torch.Tensor, *args, **kwargs):
            return self.wrapper(x, *args, **kwargs)

    return WrappedModule

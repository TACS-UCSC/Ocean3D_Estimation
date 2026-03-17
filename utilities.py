import torch
import math

def linear_beta_scheduler(timesteps, beta_start = 0.0001, beta_end = 0.02, device="cpu"):
    """
    linear schedule, proposed in original ddpm paper
    """
    # scale = 1000 / timesteps
    # beta_start = scale * beta_start
    # beta_end = scale * beta_end
    beta_start = beta_start
    beta_end = beta_end
    betas = torch.linspace(torch.tensor(beta_start), torch.tensor(beta_end), timesteps)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas.to(device), alphas.to(device), alphas_cumprod.to(device)

# def cosine_beta_scheduler(timesteps, s = 0.008, device = "cpu"):
#     """
#     cosine schedule
#     as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
#     """
#     steps = timesteps + 1
#     t = torch.linspace(0, timesteps, steps) / timesteps
#     alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
#     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#     alphas = 1 - betas
#     alphas_cumprod = torch.cumprod(alphas, dim=0) # for consistency in shape

#     return betas.to(device), alphas.to(device), alphas_cumprod.to(device)

import torch
# import jax.numpy as jnp
import gc

def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()

def numpy_to_cuda(arr):
    return torch.from_numpy(arr).float().cuda()

def cuda_to_numpy(arr):
    return arr.cpu().detach().numpy()

def count_parameters(model):
    """Counts the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cuda_memory_info():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(f"reserved: {r/10e6}")
    print(f"allocated: {a/10e6}")
    print(f"free: {f/10e6}")

n2c = numpy_to_cuda
c2n = cuda_to_numpy

def pthstr(s):
    if type(s) is int:
        return str(s).replace("-", "n").replace(".", "p")
    else:
        return str(s)


def cosine_beta_scheduler(timesteps, s=0.008, device="cpu"): # s = 0.008
    """Cosine schedule from annotated transformers.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, device=device)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.0001, 0.9500)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas.to(device), alphas.to(device), alphas_cumprod.to(device)

def sigmoid_beta_scheduler(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5, device="cpu"):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas.to(device), alphas.to(device), alphas_cumprod.to(device)


def linear_beta_scheduler2(beta_start, beta_end, timesteps, device="cpu"):
    """
    betas and alphas for the diffusion process, linear noise scheduler
    """
    betas = torch.linspace(torch.tensor(beta_start), torch.tensor(beta_end), timesteps).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device) # Precompute the cumulative product of all the alpha terms
    return betas, alphas, alphas_cumprod



# def cosine_beta_scheduler2(timesteps, s=0.008, device = "cpu", beta_clip=0.9999):
#     """
#     Generates a cosine beta schedule for the given number of timesteps.

#     Parameters:
#     - timesteps (int): The number of timesteps for the schedule.
#     - s (float): A small constant used in the calculation. Default: 0.008.
#     - device (str): The device to run the calculation on. Default: "cpu".

#     Returns:
#     - betas (torch.Tensor): The computed beta values for each timestep.
#     """
#     x = torch.arange(0, timesteps).to(device)
#     alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
#     alphas_cumprod2 = torch.roll(alphas_cumprod, 1, 0)
#     alphas_cumprod2[0] = alphas_cumprod2[1]
    
#     betas = 1 - (alphas_cumprod / alphas_cumprod2)
#     betas = torch.clip(betas,0,beta_clip).to(device)
#     alphas = 1 - betas
#     return betas, alphas, alphas_cumprod


# def cosine_beta_scheduler_old(timesteps, s=0.008, device = "cpu"):
#     """
#     Generates a cosine beta schedule for the given number of timesteps.

#     Parameters:
#     - timesteps (int): The number of timesteps for the schedule.
#     - s (float): A small constant used in the calculation. Default: 0.008.
#     - device (str): The device to run the calculation on. Default: "cpu".

#     Returns:
#     - betas (torch.Tensor): The computed beta values for each timestep.
#     """
#     steps = timesteps + 1
#     x = torch.linspace(0, timesteps, steps).to(device)
#     alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
#     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#     betas = torch.clip(betas,0,.9999).to(device)

#     alphas = 1 - betas
#     return betas, alphas, alphas_cumprod

# def cosine_beta_scheduler_test(timesteps, s=0.008, device = "cpu", x_norm_perc=1.0, beta_start=0.0001, beta_end=0.008):
#     """
#     Generates a cosine beta schedule for the given number of timesteps.

#     Parameters:
#     - timesteps (int): The number of timesteps for the schedule.
#     - s (float): A small constant used in the calculation. Default: 0.008.
#     - device (str): The device to run the calculation on. Default: "cpu".

#     Returns:
#     - betas (torch.Tensor): The computed beta values for each timestep.
#     """
#     steps = timesteps + 1 # +1 for the final timestep
#     x = torch.linspace(0, timesteps, steps).to(device)
#     alphas_cumprod = torch.cos((((x*x_norm_perc / timesteps)) + s) / (1 + s) * torch.pi * 0.5) ** 2 # goes from 1 to 0
#     #alphas_cumprod = alphas_cumprod / (alphas_cumprod[0]) # normalize to 1
#     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#     betas = torch.clip(betas,beta_start,beta_end).to(device)
#     alphas = 1 - betas
#     ## so matches the same shape as the linear scheduler
#     alphas_cumprod2 = torch.cumprod(alphas, dim=0).to(device)

#     return betas, alphas, alphas_cumprod2

def power_beta_scheduler(timesteps, beta_start=0.0001, beta_end=0.02, power=2.0, device="cpu"):
    """
    Power schedule for betas, as sometimes used in DDPM literature.
    Interpolates betas between beta_start and beta_end according to a power law.
    Args:
        timesteps (int): Number of diffusion steps.
        beta_start (float): Initial beta value.
        beta_end (float): Final beta value.
        power (float): Power/exponent for the schedule. 1.0 is linear, 2.0 is quadratic, etc.
        device (str): Torch device.
    Returns:
        betas (torch.Tensor): Beta schedule.
        alphas (torch.Tensor): 1 - betas.
        alphas_cumprod (torch.Tensor): Cumulative product of alphas.
    """
    steps = timesteps
    t = torch.linspace(0, 1, steps, device=device)
    betas = beta_start + (beta_end - beta_start) * (t ** power)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas.to(device), alphas.to(device), alphas_cumprod.to(device)

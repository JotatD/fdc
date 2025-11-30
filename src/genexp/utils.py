import torch

def sig_fn_ddpm(diff_model, t,tm1):
    """sigma_t for DDIM so that it becomes DDPM."""
    at, sig = diff_model.sde.get_alpha_sigma(t)
    atm1, _ = diff_model.sde.get_alpha_sigma(tm1)
    sig_t = torch.sqrt((1-atm1)/(1-at+1e-8)*(1-at/atm1))
    return sig_t


def noise_func(t, sqrt_alphas, sigmas):
    t = (t*999).round().long()
    return sqrt_alphas[t], sigmas[t]

def beta_t(t, beta_0, beta_1):
    """
    Continuous beta(t) = beta_0 + t*(beta_1 - beta_0).
    """
    return beta_0 + t*(beta_1 - beta_0)

# diffusion loss
def forward_noise(x, noise_func, t, eps):
    asqrt, bsqrt = noise_func(t)
    return asqrt*x + bsqrt*eps

def mse_loss(model, x, t, noise_func, eps):
    # sample random ts
    xt = forward_noise(x=x, noise_func=noise_func, t=t[:,None], eps=eps)
    # concatenate
    xt_t = torch.cat([xt, t[:,None]], dim=1)
    # pass through model
    eps_ = model(xt_t)
    # compute loss
    loss = torch.mean((eps_ - eps)**2)/2.
    return loss



def train_model(data_loader, model, optimizer, device=None):
    data_itr = iter(data_loader)
    losses = []
    for step in tqdm(range(20_000)):
        try:
            x_batch = next(data_itr).to(device)
            x_batch = x_batch.to(device)
            t_batch = torch.rand(x_batch.size(0), device=device, dtype=torch.float32)
            eps_batch = torch.randn(x_batch.size(0), 2, device=device, dtype=torch.float32)
            loss = mse_loss(model, x_batch, t_batch, noise_func, eps_batch)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        except StopIteration:
            data_itr = iter(data_loader)
            continue

def get_alpha_sigma(t, sqrt_alphas, sigmas):
    """
    Approximate alpha(t), sigma(t) by indexing into precomputed arrays
    sqrt_alphas and sigmas (each of length=1000) at index round(t*999).
    
    t: scalar in [0,1], or a PyTorch tensor in [0,1].
    Returns alpha_t (scalar), sigma_t (scalar).
    """
    # Ensure t is a PyTorch tensor
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=torch.float32)  # Convert to tensor
    if t.ndim == 0:  # Ensure it's a 1D tensor for operations
        t = t.view(1)
    
    # Compute index (rounding and clamping to [0, 999])
    idx = torch.clamp(torch.round(t * 999), min=0, max=999).long()  # Ensure index is an integer tensor
    
    # Fetch alpha and sigma
    alpha_t = sqrt_alphas[idx].item()**2  # alpha(t), scalar
    sigma_t = sigmas[idx].item()          # sigma(t), scalar
    return alpha_t, sigma_t


def pf_ode(t, x, model):
    """Probability flow ODE."""

    device = x.device
    # Ensure t is a 1D tensor so we can broadcast
    if not torch.is_tensor(t):
        t = torch.tensor([t], dtype=x.dtype, device=device)
    elif t.ndim == 0:
        t = t.view(1)

    # 1) Compute alpha_t, sigma_t by indexing
    alpha_t, sigma_t = get_alpha_sigma(t.item())  # scalar
    # or if you have a batch version, you'd do something vectorized.

    # 2) Evaluate beta(t)
    b_t = beta_t(t)  # shape (1,) if t is shape (1,)

    # 3) Get eps_pred from the model
    #    Typically: model( concat[x, t], ) => shape (batch, dim)
    #    You might do something like:
    t_expand = t.repeat(x.size(0), 1)  # shape (batch, 1)
    inp = torch.cat([x, t_expand], dim=1)  # shape (batch, dim+1)
    eps_pred = model(inp)  # shape (batch, dim)

    # 4) Convert eps_pred => score factor: (+0.5 * beta(t) * eps_pred / sigma_t)
    drift = -0.5 * b_t * x  # shape (batch, dim)
    score_term = 0.5 * b_t * eps_pred / (sigma_t + 1e-12)  # avoid div-by-zero
    dx_dt = drift + score_term

    return dx_dt




"""Divergence estimators."""

def skilling_hutchinson_divergence(x, f, eps=None, dim=[1]):
    """Compute the divergence with Skilling-Hutchinson for f(x)."""
    if eps is None:
        eps = torch.randn_like(x)
    with torch.enable_grad():
      out = torch.sum(f * eps)
      grad_x_f = torch.autograd.grad(out, x, retain_graph=True)[0]
    return torch.sum(grad_x_f * eps, dim=dim)    
  

import numpy as np
def discrete_entropy(counts):
    total = counts.sum()
    pxy = counts / total 
    p_nonzero = pxy[pxy > 0]
    H_bits = -np.sum(p_nonzero * np.log2(p_nonzero))
    H_nats = -np.sum(p_nonzero * np.log(p_nonzero))
    return H_nats






"""DDIM solver"""
def ddim_step(x, t, tm1, model, noise_func, device='cuda'):
    # model prediction
    x = x.to(device)

    t_in = torch.tensor(t, device=device).expand(x.size(0))[:, None]

    eps_pred = model(torch.cat([x, t_in], dim=1))

    avoid_inf = 1e-6

    atm1, btm1 = noise_func(tm1)
    at, bt = noise_func(t)

    x0_pred = (x - bt*eps_pred)/(at + avoid_inf)
    #print(eps_pred.shape, x.shape, x0_pred.shape)
    xtm1 = atm1 * x0_pred + btm1 * eps_pred
    return xtm1

def ode_solver(x0, step_func, ts=None, steps=50,store_traj=False):
    if ts is None:
        ts = torch.linspace(1,0, steps+1)
    if store_traj:
        traj = []
    else:
        traj = None
    xt = x0
    for t, tm1 in zip(ts[:-1], ts[1:]):
        xt = step_func(x=xt, t=t, tm1=tm1)
        if traj is not None:
            traj.append(xt.cpu().detach())
    return xt, {'traj': traj}

def cast_to_half(module: torch.nn.Module):
    """
    Recursively casts all float parameters and buffers in `module` to half precision.
    
    Args:
        module (nn.Module): The PyTorch module whose float parameters should be cast to FP16.
    """
    # First, cast parameters in the current module (no recursion here)
    for param in module.parameters(recurse=False):
        if param.dtype == torch.float32:
            param.data = param.data.half()

    # Cast buffers (e.g., running averages in BatchNorm)
    for name, buffer in module.named_buffers(recurse=False):
        if buffer.dtype == torch.float32:
            module.register_buffer(name, buffer.half())

    # Recursively apply to child modules
    for child in module.children():
        cast_to_half(child)


def recursive_to_device(module: torch.nn.Module, device: torch.device) -> None:
    """
    Recursively casts all parameters, their gradients (if any), and buffers
    within the given module and its submodules to the specified device.
    
    Args:
        module (nn.Module): The PyTorch module whose parameters and buffers
                            will be moved.
        device (torch.device): The target device (e.g., torch.device("cuda"),
                               torch.device("cpu"), etc.).
    """
    
    # Move parameters and their gradients to the specified device
    for param in module.parameters(recurse=False):
        # Move the parameter itself
        param.data = param.data.to(device)
        # If there is a gradient, move it as well
        if param.grad is not None:
            param.grad.data = param.grad.data.to(device)

    # Move all registered buffers (e.g., running averages in BatchNorm, etc.) 
    for key in module._buffers:
        buffer = module._buffers[key]
        if buffer is not None:
            module._buffers[key] = buffer.to(device)
    
    # Recursively apply to submodules
    for child in module.children():
        recursive_to_device(child, device)




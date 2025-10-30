import torch

def get_moments(x: torch.Tensor) -> torch.Tensor:
    # handle shape (n, t) -> (1, t)
    if x.dim() == 2:
        x_obs = x[0, :].unsqueeze(0)
    # handle shape (n, b, t) -> (b, t)
    elif x.dim() == 3:
        x_obs = x[0, :, :]
    # handle shape (t) for real data -> (1, t)
    else:
        x_obs = x.unsqueeze(0)
    mean = x_obs.mean(dim=-1)
    var = x_obs.var(dim=-1)
    std = var.sqrt()
    norm = (x_obs - mean.unsqueeze(-1)) / std.unsqueeze(-1)
    skew = (norm**3).mean(dim=-1)
    kurt = (norm**4).mean(dim=-1)
    return torch.stack((mean, var, skew, kurt), dim=-1).to(x.device) # shape: (4, b, 1)

from typing import Any, Sequence

import numpy as np
import torch
from tqdm import tqdm


class Solver():
    def __init__(self):
        def implicit_euler(sde: Any, x0: torch.Tensor, ts: torch.Tensor, max_iter: int = 10, tol: float = 1e-6) -> torch.Tensor:
            """
            Implicit Euler-Maruyama SDE solver
            :param sde: SDE class containing drift and diffusion functions
            :param x0: initial conditions; shape: (batch size, d)
            :param ts: times to evaluate at; shape: (n)
            :param dt: time step
            :param max_iter: maximum number of iterations when performing the update step x_{n+1} = x_n + f(t_{n+1}, x_{n+1})dt + g(t_n, x_n)dW
            :param tol: tolerance to check for convergence
            :return: tensor of the solution; shape: (n, batch size, d)
            """
            # create tensor of the solution
            batch_size, d = x0.shape
            n = ts.shape[0]
            xs = torch.zeros((n, batch_size, d), dtype=x0.dtype, device=x0.device)
            xs[0] = x0

            # time and state independent drift
            g = sde.g()

            # recursively define x_{n+1}
            for i in tqdm(range(0, n-1), desc=f"Simulating {batch_size} batches of hair bundles"):
                t_curr, t_next = ts[i], ts[i+1]
                dt = t_next.item() - t_curr.item()
                x_curr = xs[i]
                # Wiener process
                dW = torch.rand_like(x_curr) * np.sqrt(dt)
                eta = torch.bmm(g, dW.unsqueeze(-1)).squeeze(-1) # batch matrix multiplication; shape: (batch_size, d)
                # recursive iteration
                x_next = x_curr.clone() # possible candidate for solution at next time step
                for _ in range(max_iter):
                    x_temp = x_curr + sde.f(x_next, t_next) * dt + eta
                    if torch.norm(x_temp - x_next) < tol:
                        break # convergence check
                    x_next = x_temp
                # update solution
                xs[i+1] = x_next

            return xs

        self.implicit_euler = implicit_euler
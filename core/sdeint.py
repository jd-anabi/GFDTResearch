import math
from typing import Any, Sequence

import torch
from tqdm import tqdm

class Solver:
    def __init__(self):
        def euler(sde: Any, x0: torch.Tensor, ts: tuple[float, float], n: int) -> torch.Tensor:
            """
            Explicit Euler-Maruyama SDE solver
            :param sde: SDE class containing drift and diffusion functions
            :param x0: initial conditions; shape: (batch size, d)
            :param ts: time span to solve SDEs for
            :param n: number of time steps
            :return: tensor of the solution; shape: (n, batch size, d)
            """
            # time info
            t = torch.linspace(*ts, n)
            dt = t[1].item() - t[0].item() # fixed time step

            # dimensions
            batch_size, d = x0.shape

            # initialize solution array
            xs = torch.zeros((n, batch_size, d), dtype=x0.dtype, device=x0.device)
            xs[0] = x0

            # drift
            g = sde.g()

            # pre-compute constants
            sqrt_dt = math.sqrt(dt)

            # recursively define x_{n+1}
            for i in tqdm(range(0, n - 1),  desc=f"Simulating {batch_size} batches of hair-bundles", mininterval=0.1):
                t_curr, t_next = t[i], t[i + 1]
                x_curr = xs[i]
                # Wiener process
                dW = torch.randn_like(x_curr) * sqrt_dt
                eta = torch.bmm(g, dW.unsqueeze(-1)).squeeze(-1)  # batch matrix multiplication; shape: (batch_size, d)
                # update solution
                a = sde.f(x_curr, t_curr) * dt
                xs[i + 1] = x_curr + sde.f(x_curr, t_curr) * dt + eta

            return xs

        def implicit_euler(sde: Any, x0: torch.Tensor, ts: tuple[float, float], n: int, max_iter: int = 10, tol: float = 1e-6) -> torch.Tensor:
            """
            Implicit Euler-Maruyama SDE solver
            :param sde: SDE class containing drift and diffusion functions
            :param x0: initial conditions; shape: (batch size, d)
            :param ts: times span to solve SDEs for
            :param n: number of time steps
            :param max_iter: maximum number of iterations when performing the update step x_{n+1} = x_n + f(t_{n+1}, x_{n+1})dt + g(t_n, x_n)dW
            :param tol: tolerance to check for convergence
            :return: tensor of the solution; shape: (n, batch size, d)
            """
            # time info
            t = torch.linspace(*ts, n)
            dt = t[1].item() - t[0].item() # fixed time step

            # create tensor of the solution
            batch_size, d = x0.shape
            xs = torch.zeros((n, batch_size, d), dtype=x0.dtype, device=x0.device)
            xs[0] = x0

            # time and state independent drift
            g = sde.g()

            # pre-compute constants
            sqrt_dt = math.sqrt(dt)

            # recursively define x_{n+1}
            for i in tqdm(range(0, n - 1),  desc=f"Simulating {batch_size} batches of hair-bundles", mininterval=0.1):
                t_curr, t_next = t[i], t[i+1]
                x_curr = xs[i]
                # Wiener process
                dW = torch.randn_like(x_curr) * sqrt_dt
                eta = torch.bmm(g, dW.unsqueeze(-1)).squeeze(-1) # batch matrix multiplication; shape: (batch_size, d)
                # recursive iteration
                x_next = x_curr.clone() # possible candidate for the solution at next time step
                for _ in range(max_iter):
                    x_temp = x_curr + sde.f(x_next, t_next) * dt + eta
                    if torch.norm(x_temp - x_next) < tol:
                        break # convergence check
                    x_next = x_temp
                # update solution
                xs[i+1] = x_next

            return xs

        self.euler = euler
        self.implicit_euler = implicit_euler
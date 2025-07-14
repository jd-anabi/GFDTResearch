import numpy as np
import torch

BOLTZMANN_RESCALE = 1e18 # nm^2 mg ms^-2 K^-1
K_B = BOLTZMANN_RESCALE * 1.380649e-23 # m^2 kg s^-2 K^-1

class HarmonicOscillator(torch.nn.Module):
    def __init__(self, mass: float, gamma: float, omega_0: float, temp: float,
                 omega: np.ndarray, amp: float, phase: np.ndarray, offset: float,
                 batch_size: int, device: torch.device = 'cuda', dtype: torch.dtype = torch.float64):
        super().__init__()
        # parameters
        self.mass = mass
        self.gamma = gamma
        self.omega_0 = omega_0
        self.temp = temp
        self.eta = np.sqrt(2 * K_B * self.gamma * self.temp / self.mass)

        # force parameters
        self.amp = amp  # amplitude of stimulus force
        self.phase = torch.tensor(phase, dtype=dtype, device=device)  # phase of the stimulus force
        self.offset = offset  # stimulus force offset
        self.omega = torch.tensor(omega, dtype=dtype, device=device)

        # other parameters
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

    def f(self, x, t) -> torch.Tensor:
        dx = x[:, 1]
        dv = -1 * self.omega_0 * x[:, 0] - self.gamma / self.mass * x[:, 1] + self.__sin_sf(t) / self.mass
        dx = torch.stack((dx, dv), dim=1)
        return dx

    def g(self, x: torch.tensor = None, t: torch.tensor = None) -> torch.Tensor:
        dsigma = torch.tensor([0, self.eta], dtype=self.dtype, device=self.device)
        return torch.tile(torch.diag(dsigma), (self.batch_size, 1, 1))

    # stimulus force
    def __sin_sf(self, t):
        return self.amp * torch.sin(self.omega * t + self.phase) + self.offset
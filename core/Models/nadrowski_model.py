import torch


@torch.jit.script
def _nadrowski_compiled_step(
    x: torch.Tensor, force_step: torch.Tensor, dW: torch.Tensor,
    k: torch.Tensor, lam: torch.Tensor, f_max: torch.Tensor, tau: torch.Tensor,
    tau_c: torch.Tensor, s: torch.Tensor, a: torch.Tensor, beta: torch.Tensor,
    n: torch.Tensor, temp: torch.Tensor,
    x_noise: torch.Tensor, y_noise: torch.Tensor,
    dt: float, sqrt_dt: float,
) -> torch.Tensor:
    """One Euler-Maruyama step for Nadrowski (state-dependent diffusion), JIT-scripted.

    x_noise / y_noise arrive PRECOMPUTED (see NadrowskiModel.__init__): they are pure functions of
    the parameter tensors, and torch.jit.script cannot hoist them out because those tensors are
    function inputs -- so they were two torch.sqrt calls on every one of ~2.4M steps per segment.
    They are APPENDED to the signature rather than replacing n/beta/temp: the solver splats
    compiled_params() positionally, so reordering would silently mis-bind parameters. `temp` is now
    unused here and is kept only to preserve those positions.
    """
    x0 = x[:, 0]
    x1 = x[:, 1]
    x2 = x[:, 2]
    p = 1.0 / (1.0 + a * torch.exp(-beta * (x0 - x1)))
    x_gs = x0 - x1 - p
    dx = -(x_gs + k * x0) + force_step[:, 0]
    dy = (x_gs - f_max * (1 - s * x2)) / lam
    dc = (p - x2) / tau
    drift = torch.stack((dx, dy, dc), dim=1)
    c_noise = torch.sqrt(2.0 * tau_c * p * (1 - p) / n) / tau
    g = torch.stack((x_noise, y_noise, c_noise), dim=-1)
    return x + drift * dt + g * dW * sqrt_dt


class NadrowskiModel:
    compiled_step = staticmethod(_nadrowski_compiled_step)

    def __init__(self, k: torch.Tensor, lam: torch.Tensor, f: torch.Tensor, tau: torch.Tensor, tau_c: torch.Tensor,
                 s: torch.Tensor, delta_e: torch.Tensor, beta: torch.Tensor, n: torch.Tensor, temp: torch.Tensor,
                 force: torch.Tensor, batch_size: int, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32):
        # sde model parameters
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

        # parameters
        self.k = k.to(self.device)
        self.lam = lam.to(self.device)
        self.f_max = f.to(self.device)
        self.tau = tau.to(self.device)
        self.tau_c = tau_c.to(self.device)
        self.s = s.to(self.device)
        self.delta_e = delta_e.to(self.device)
        self.beta = beta.to(self.device)
        self.n = n.to(self.device)
        self.temp = temp.to(self.device)

        # force parameters
        self.force = force.to(self.device)

        # subsuming parameters
        self.a = torch.exp(self.delta_e + self.beta / 2)

        # Constant noise amplitudes. Pure functions of the parameter tensors, so they are computed
        # ONCE here rather than on every solver step: Nadrowski runs the state_dep_drift branch, so
        # g(x) is called per Euler step -- up to ~2.4M times per segment, x segments x runs x 5000
        # training batches. Only c_noise genuinely depends on the state.
        self._x_noise_const = torch.sqrt(2 / (self.n * self.beta))
        self._y_noise_const = torch.sqrt(2 * self.temp / (self.n * self.beta * self.lam))

    def f(self, x, t) -> torch.Tensor:
        return self.f_pure(x, self.force[:, :, t])

    def f_pure(self, x: torch.Tensor, force_step: torch.Tensor) -> torch.Tensor:
        """Drift as a pure function of state and a pre-sliced force vector."""
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        p = self.__p_t0(x0, x1)
        x_gs = x0 - x1 - p
        dx = -(x_gs + self.k * x0) + force_step[:, 0]
        dy = (x_gs - self.f_max * (1 - self.s * x2)) / self.lam
        dc = (p - x2) / self.tau
        return torch.stack((dx, dy, dc), dim=1)

    def compiled_params(self) -> tuple:
        """Tuple of tensor params for `compiled_step` (after x, force_step, dW).

        ORDER IS LOAD-BEARING -- the solver splats this positionally into compiled_step, and every
        entry is a same-shaped tensor, so a reordering mis-binds parameters silently (wrong physics,
        no error). Only ever APPEND.
        """
        return (self.k, self.lam, self.f_max, self.tau, self.tau_c,
                self.s, self.a, self.beta, self.n, self.temp,
                self._x_noise_const, self._y_noise_const)

    def g(self, x) -> torch.Tensor:
        # Diagonal noise as a (batch, d) vector; solver multiplies elementwise.
        p = self.__p_t0(x[:, 0], x[:, 1])
        c_noise = torch.sqrt(2 * self.tau_c * p * (1 - p) / self.n) / self.tau
        return torch.stack((self._x_noise_const, self._y_noise_const, c_noise), dim=-1)

    # --- NOISE --- #
    # Kept as accessors for any external caller; both are precomputed in __init__ (see there).
    def _x_noise(self) -> torch.Tensor:
        return self._x_noise_const

    def _y_noise(self) -> torch.Tensor:
        return self._y_noise_const

    # --- PRIVATE --- #
    def __p_t0(self, x, y):
        return 1 / (1 + self.a * torch.exp(-1 * self.beta * (x - y)))
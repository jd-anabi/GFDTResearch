import numpy as np

from hair_bundle import HairBundle

class HairBundleSDE(HairBundle):
    def __init__(self, params: list, force_params: list, p_t_steady_state: bool):
        super().__init__(*params, *force_params, p_t_steady_state)

        def f(dxdt: list, x: list, p: list, t: float) -> None:
            """
            Deterministic part of the coupled SDEs
            :param x: list of the n variables of the coupled SDEs
            :param t: the time to evaluate the coupled SDEs at
            :return: an array of length n representing the system of the coupled SDEs at time t
            """
            x_sf = self.sin_driving_force(t)
            dxdt = np.array(self.sde_sym_lambda_func(x_sf + x[0], *x[1:]))

        def g(sigma: list, x: list, p: list, t: float) -> None:
            """
            Noisy part of the coupled SDEs
            :param x: list of the n variables of the coupled SDEs
            :param t: the time to evaluate the noise at; not needed because the noise for the hair bundle is time-independent
            :return: list of length n representing the noise associated with each variable
            """
            sigma = [self.hb_noise, self.a_noise, 0, 0, 0]

        self.f = f
        self.g = g
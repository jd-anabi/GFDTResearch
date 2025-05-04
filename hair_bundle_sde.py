import numpy as np

from hair_bundle import HairBundle

class HairBundleSDE(HairBundle):
    def __init__(self, params: list, force_params: list, p_t_steady_state: bool):
        super().__init__(*params, *force_params, p_t_steady_state)

        def f(x: list, t: float) -> np.ndarray:
            """
            Deterministic part of the coupled SDEs
            :param x: list of the n variables of the coupled SDEs
            :param t: the time to evaluate the coupled SDEs at
            :return: an array of length n representing the system of the coupled SDEs at time t
            """
            x_sf = self.driving_force(t)
            sde_sys = np.array(self.sde_sym_lambda_func(x_sf + x[0], *x[1:]))
            return sde_sys

        def g(x: list, t: float) -> list:
            """
            Noisy part of the coupled SDEs
            :param x: list of the n variables of the coupled SDEs
            :param t: the time to evaluate the noise at; not needed because the noise for the hair bundle is time-independent
            :return: list of length n representing the noise associated with each variable
            """
            noise = [self.hb_noise, self.a_noise, 0, 0, 0]
            return noise[0:len(x)]

        self.f = f
        self.g = g
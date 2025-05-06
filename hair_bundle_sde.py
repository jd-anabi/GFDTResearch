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
            #x_sf = self.sin_driving_force(t)
            u = self.sde_sym_lambda_func(x[0], *x[1:])
            dxdt[0] = u[0]
            dxdt[1] = u[1]
            dxdt[2] = u[2]
            dxdt[3] = u[3]

        def g(sigma: list, x: list, p: list, t: float) -> None:
            """
            Noisy part of the coupled SDEs
            :param x: list of the n variables of the coupled SDEs
            :param t: the time to evaluate the noise at; not needed because the noise for the hair bundle is time-independent
            :return: list of length n representing the noise associated with each variable
            """
            du = [self.hb_noise, self.a_noise, 0, 0, 0]
            sigma[0] = du[0]
            sigma[1] = du[1]
            sigma[2] = du[2]
            sigma[3] = du[3]

        self.f = f
        self.g = g
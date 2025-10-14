import numpy as np

def rescale_x(x_nd: np.ndarray, gamma: float, d: float, x_sp: float, chi_hb: float) -> np.ndarray:
    """
    Rescaling the hair-bundle displacement
    :param x_nd: the hair bundle position
    :param gamma: geometric conversion factor
    :param d: distance of gating spring relaxation on channel opening
    :param x_sp: resting deflection of stereociliary pivots
    :param chi_hb: non-dimensional parameter for non-dimensional hair bundle displacement
    :return: the rescaled hair-bundle displacement
    """
    x = chi_hb * d / gamma * x_nd + x_sp
    return x

def irescale_x(x: float, gamma: float, d: float, x_sp: float, chi_hb: float) -> float:
    """
    Rescaling the hair-bundle displacement from dimensional -> non-dimensional
    :param x: the hair bundle position
    :param gamma: geometric conversion factor
    :param d: distance of gating spring relaxation on channel opening
    :param x_sp: resting deflection of stereociliary pivots
    :param chi_hb: non-dimensional parameter for non-dimensional hair bundle displacement
    :return: the rescaled hair-bundle displacement
    """
    x_nd = gamma * (x - x_sp) / (chi_hb * d)
    return x_nd

def rescale_t(nd_t: np.ndarray, k_gs_max: float, s_max: float, t_0: float, s_max_nd: float, chi_a: float) -> np.ndarray:
    """
    Rescaling the time array
    :param nd_t: the time array
    :param k_gs_max: maximum stiffness of gating spring
    :param s_max: maximum slipping rate
    :param t_0: time offset
    :param s_max_nd: non-dimensional maximum slipping rate
    :param chi_a: non-dimensional parameter for non-dimensional adaptation motor displacement
    :return:  the rescaled time
    """
    t = chi_a * s_max_nd / (k_gs_max * s_max) * nd_t - t_0
    return t

def irescale_f(force: np.ndarray, gamma: float, d: float, k_sp: float, chi_hb: float) -> np.ndarray:
    """
    Rescaling the stimulus force from dimensional -> non-dimensional
    :param force: the stimulus force position
    :param gamma: geometric conversion factor
    :param d: distance of gating spring relaxation on channel opening
    :param k_sp: stiffness of stereociliary pivots
    :param chi_hb: non-dimensional parameter for non-dimensional hair bundle displacement
    :return: the rescaled stimulus force
    """
    force_nd = gamma / (chi_hb * k_sp * d) * force
    return force_nd

def irescale_f_params(omegas: np.ndarray, amp: float, phase: float, offset: float,
                      gamma: float, d: float, k_sp: float, chi_hb: float,
                      k_gs_max: float, s_max: float, s_max_nd: float, chi_a: float, t_0: float) -> tuple[np.ndarray, float, np.ndarray, float]:
    """
    Rescale the stimulus force (sinusoidal force)  parameters from dimensional -> non-dimensional
    :param amp: amplitude
    :param omegas: frequencies
    :param phase: phase
    :param offset: offset
    :param gamma: geometric conversion factor
    :param d: distance of gating spring relaxation on channel opening
    :param k_sp: stiffness of stereociliary pivots
    :param chi_hb: non-dimensional parameter for non-dimensional hair bundle displacement
    :param k_gs_max: maximum stiffness of gating spring
    :param s_max: maximum slipping rate
    :param s_max_nd: non-dimensional maximum slipping rate
    :param chi_a: non-dimensional parameter for non-dimensional adaptation motor displacement
    :param t_0: time offset
    :return: the rescaled stimulus force parameters
    """
    alpha = gamma / (chi_hb * k_sp * d)
    t_prime = chi_a * s_max_nd / (k_gs_max * s_max)
    amp_nd = alpha * amp
    offset_nd = alpha * offset
    omega_nd = t_prime * omegas
    phases_nd = phase - t_0 * omegas
    return omega_nd, amp_nd, phases_nd, offset_nd
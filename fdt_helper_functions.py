import numpy as np
import scipy as sp

def __correlation(trials: np.ndarray, num_time_steps: int) -> np.ndarray:
    c = np.zeros(num_time_steps)
    for t in range(num_time_steps):
        for i in range(len(trials)):
            c[t] += trials[i][t] * trials[i][0]
        c[t] /= num_time_steps
    return c

def __linear_response_freq(hb_disp: list, f_a: list, f_hb0: float) -> np.ndarray:
    hb_disp_freq = sp.fft.fft(hb_disp)
    hb_acc_freq = sp.fft.fft([f_a_t + f_hb0 for f_a_t in f_a])
    return np.array([hb_disp_freq[i] / hb_acc_freq[i] for i in range(len(hb_disp))])

def fdt_test(hb_test_runs: np.ndarray, num_steps: int, time_step: float, f_a: np.ndarray, f_hb0: float) -> tuple:
    ratio = np.zeros(num_steps, dtype=complex)
    omega = sp.fft.fftfreq(num_steps, time_step)
    omega = sp.fft.fftshift(omega) # type: np.ndarray
    c = __correlation(hb_test_runs, num_steps)
    c_freq = sp.fft.fft(c)
    c_freq = sp.fft.fftshift(c_freq) # type: np.ndarray
    chi_freq = __linear_response_freq(np.mean(hb_test_runs, axis=0), np.mean(f_a, axis=0), f_hb0)
    chi_freq = sp.fft.fftshift(chi_freq) # type: np.ndarray
    for i in range(len(chi_freq)):
        ratio[i] = omega[i] * c_freq[i] / (chi_freq[i].imag + 1e-9)
    return omega, ratio
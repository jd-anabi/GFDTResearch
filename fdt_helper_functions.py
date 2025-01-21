import math
import numpy as np
import scipy as sp

def __correlation(trials: list, num_time_steps: int) -> np.ndarray:
    c = np.zeros(num_time_steps)
    for t in range(num_time_steps):
        for i in range(len(trials)):
            c[t] += trials[i][t] * trials[i][0]
        c[t] /= num_time_steps
    return c

def __linear_response_freq(hb_disp: list) -> np.ndarray:
    hb_acc = np.gradient(np.gradient(hb_disp))
    hb_disp_freq = sp.fft.fft(hb_disp)
    hb_acc_freq = sp.fft.fft(hb_acc)
    return np.array([hb_disp_freq[i] / hb_acc_freq[i] for i in range(len(hb_disp))])

def fdt_test(hb_test_runs: list, max_time: int, time_step: float, temp: float) -> bool:
    fdt_status = True
    ratio = 1
    omega = sp.fft.fftfreq(max_time, time_step)
    omega = sp.fft.fftshift(omega)
    c = __correlation(hb_test_runs, len(hb_test_runs[0]))
    c_freq = sp.fft.fft(c)
    c_freq = sp.fft.fftshift(c_freq)
    chi_freq = __linear_response_freq(hb_test_runs[0])
    chi_freq = sp.fft.fftshift(chi_freq)
    for i in range(len(chi_freq)):
        ratio = omega[i] * chi_freq[i] / (2 * sp.constants.k * temp * chi_freq[i].imag)
        if not math.isclose(ratio, 1):
            fdt_status = False
    return fdt_status
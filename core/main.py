import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from core import helpers as helpers

if __name__ == '__main__':
    # time and frequency arrays
    dt = 1e-6
    ts = (0, 10)
    n = int((ts[-1] - ts[0]) / dt)
    t = np.linspace(ts[0], ts[-1], n)
    freq = sp.fft.fftshift(sp.fft.fftfreq(len(t), dt))[len(t) // 2:]

    # parameters
    nd = True
    file_str = 'Dimensional' if not nd else 'Non-dimensional'
    params = np.zeros(33, dtype=float) if not nd else np.zeros(17, dtype=float)

    # read hair cell and force info from txt files
    line = 0
    if sys.platform == 'win32':
        hair_cell_file_path = '\\Hair Cells\\' + file_str + '\\hair_cell_'
        force_file_path = '\\Force Info\\sin_force_params.txt'
    else:
        hair_cell_file_path = '/Hair Cells/' + file_str + '/hair_cell_'
        force_file_path = '/Force Info/sin_force_params.txt'
    file = os.getcwd() + hair_cell_file_path + input('Hair cell file number: ') + '.txt'
    pattern = re.compile(r'[\s=]+([+-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?)$')  # use pattern matching to extract values
    x0 = []
    with open(file, mode='r') as txtfile:
        for row in txtfile:
            val = float(re.findall(pattern, row.strip())[0])
            if line < 5:
                x0.append(val)
            else:
                params[line - 5] = val
            line = line + 1
    print(params)

    # read user input for spontaneous oscilaltion frequency and whether to use the steady-state solution for the open-channel probability
    osc_freq_center = 2 * sp.constants.pi * float(input("Frequency to center driving at (Hz): "))
    pt0_params = []

    # check if we want to use the steady-state solution
    if params[0] == 0:
        pt0_params = [params[5], *params[10:14], params[31]]

    # set up forcing params
    file = os.getcwd() + force_file_path
    forcing_amps = []
    with open(file, mode='r') as txtfile:
        for row in txtfile:
            val = float(re.findall(pattern, row.strip())[0])
            forcing_amps.append(val)
    amp = forcing_amps[0]
    k_sf = forcing_amps[1]
    args_list = (ts, n, x0, list(params), [osc_freq_center, amp, k_sf], nd)

    # multiprocessing and solve sdes
    results = helpers.hb_sols(*args_list) # shape: (T, BATCH_SIZE, d)
    print(results)

    # separate driven and not driven data
    hb_pos0 = results[:, 0, 0]  # data of the hair bundle position from the first batch
    print(hb_pos0)

    # get frequency of spontaneous oscillations
    hb_pos0_freq = sp.fft.fftshift(sp.fft.fft(hb_pos0 - np.mean(hb_pos0)))[len(t) // 2:]  # fft for non-driven data
    #spon_osc_freq = freq[np.argmax(hb_pos0_freq)]  # frequency of spontaneous oscillations
    #print(f'Frequency of spontaneous oscillations: {spon_osc_freq} Hz. Angular frequency: {2 * np.pi * spon_osc_freq} rad/s')

    # preliminary plotting
    plt.plot(t, hb_pos0)
    plt.xlabel(r'Time')
    plt.ylabel(r'$x_{hb}$')
    plt.show()

    plt.plot(t, results[:, 0, 4])
    plt.xlabel(r'Time')
    plt.ylabel(r'$x_{hb}$')
    plt.show()
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import helpers

if __name__ == '__main__':
    # time and frequency arrays
    t = np.linspace(0, 1, int(1e4))
    dt = float(t[1] - t[0])
    freq = sp.fft.fftshift(sp.fft.fftfreq(len(t), dt))[len(t) // 2:]

    # parameters
    params = np.zeros(33, dtype=float)

    # read hair cell and force info from txt files
    line = 0
    if sys.platform == 'win32':
        hair_cell_file_path = '\\Hair Cells\\hair_cell_'
        force_file_path = '\\Force Info\\sin_force_params.txt'
    else:
        hair_cell_file_path = '/Hair Cells/hair_cell_'
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
    vis_amp = forcing_amps[1]
    args_list = (t, x0, list(params), [0, amp, vis_amp])

    # multiprocessing and solve sdes
    results = helpers.hb_sols(*args_list) # shape: (T, BATCH_SIZE, d)
    print(results)

    # separate driven and not driven data
    hb_pos0 = results[:, 0, 0]  # data of the hair bundle position from the first batch
    print(hb_pos0)
    print(np.argwhere(np.isnan(hb_pos0)))

    # get frequency of spontaneous oscillations
    hb_pos0_freq = sp.fft.fftshift(sp.fft.fft(hb_pos0 - np.mean(hb_pos0)))[len(t) // 2:]  # fft for non-driven data
    spon_osc_freq = freq[np.argmax(hb_pos0_freq)]  # frequency of spontaneous oscillations
    print(f'Frequency of spontaneous oscillations: {spon_osc_freq} Hz. Angular frequency: {2 * np.pi * spon_osc_freq} rad/s')

    # preliminary plotting
    plt.plot(t, hb_pos0)
    plt.xlabel(r'Time')
    plt.ylabel(r'$x_{hb}$')
    plt.show()

    plt.plot(hb_pos0)
    plt.xlabel(r'Time')
    plt.ylabel(r'$x_{hb}$')
    plt.show()
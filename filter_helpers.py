"""
Helper functions for the filter optimization

functions written by: David Ramsay
"""
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import colorConverter
import math


def tfvar_to_zpk(zeros, poles):
    """
    Takes in a lists of zeros and poles as tf.Variable and converts them to
    complex poles and zeros
    """
    
    zs = [z.numpy()[0] for z in zeros]
    ps = []
    for p in poles:
        ps.append(p.numpy()[0] + 1j*p.numpy()[1])
        ps.append(p.numpy()[0] - 1j*p.numpy()[1])

    return zs, ps


def get_complex_targ(sos, freqs, fs):
    #return db mag and phase at frequencies given an sos filter
    return sig.sosfreqz(sos, 2.0*np.pi*np.array(freqs)/fs, fs)


def get_mag_targ(sos, freqs, fs):
    #return db mag at radian angles given an sos filter
    w, h = sig.sosfreqz(sos, 2.0*np.pi*np.array(freqs)/fs, fs)
    return w, 20*np.log10(np.abs(h))


def octaves_to(freq, appendFs=False):
    #get octave freqs from 20 hz to <fs, +fs
    octspace = [20]

    while octspace[-1]*2.0 <= freq: octspace.append(int(octspace[-1]*2.0))
    if octspace[-1] != freq and appendFs: octspace.append(int(freq))

    return octspace

# def plot_sos(sos, fs, show=True):


def plot_zpk(z, p, k, worN=5000, fs=44.1e3, show=True):

    if isinstance(worN, float): 
        print("Float given instead of integer for length of freqz. This will be converted to an integer")
        worN=int(worN)

    # int casting to ensure supplied the number of frequencies to calculate
    w, h = sig.freqz_zpk(z, p, k, worN=worN) # rad/samp, magnitudes

    fig = plt.figure(figsize=(20,4))
    gs = GridSpec(2, 2, width_ratios = [1,2])

    #plot poles/zeros
    ax1 = fig.add_subplot(gs[0:,0])

    #gives the unit circle
    unit_circle = patches.Circle((0,0), radius=1, fill=False,
                                 color='grey', ls='dotted', alpha=0.5)
    ax1.add_patch(unit_circle)

    # plot poles and zeros
    for pole in p:
        plt.plot(pole.real, pole.imag, 'x', markersize=7, alpha=0.7, color='blue')
    for zero in z:
        plt.plot(zero.real, zero.imag, 'o', markersize=7, markerfacecolor='none',
                markeredgecolor=colorConverter.to_rgba('mediumseagreen', alpha=0.7))
    ax1.grid(True)
    ax1.set_xlim(-1.2,1.2)
    ax1.set_ylim(-1.2,1.2)
    ax1.set_title('Z domain Poles/Zeros')

    #plot mag/phase
    ax2 = fig.add_subplot(gs[0,1:])
    db = 20*np.log10(np.abs(h))
    ax2.plot(w/np.pi*fs/2, db)
    ax2.set_xscale('symlog')
    spacing = octaves_to(fs/2)
    ax2.set_xticks(spacing)
    ax2.set_xticklabels(spacing, rotation=30)
    ax2.set_ylim(-75, 5)
    ax2.grid(True)
    ax2.set_yticks([0, -20, -40, -60])
    ax2.set_ylabel('Gain [dB]')
    ax2.set_title('Frequency Response')

    ax3 = fig.add_subplot(gs[1,1:])
    ax3.plot(w/np.pi*fs/2, np.angle(h))
    ax3.set_xscale('symlog')
    ax3.set_xticks(spacing)
    ax3.set_xticklabels(spacing, rotation=30)
    ax3.grid(True)
    ax3.set_yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],
             [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    ax3.set_ylabel('Phase [rad]')
    ax3.set_xlabel('Frequency (Hz)')

    if show: plt.show()



# def plot_zpk(z, p, k, fs, show=True):
#     plot_sos(sig.zpk2sos(z,p,k), fs, show)

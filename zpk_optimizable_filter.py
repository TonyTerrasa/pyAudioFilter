"""
Class that trains biquad filter by adjusting the poles and zeros to get a desires
amplitude response. Note that there is no preservation of phase at this point and that
loss is only calculated against the give frequency and amplitudes


@author: Tony Terrasa
based off of work and guidance of David Ramsay
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal as sig
from pythonAudioMeasurements.audioSample import audioSample
from pyAudioFilter.filter_helpers import *


#plot_sos(target, FS)

MAX_POLE_NORM = 0.900

class ZPKOptimizableFilter:

    def __init__(self, num_zeros=2, num_poles=1, learning_rate=0.1):
        # optimizing/pushing the poles and zeros of this system
        self.zs = [tf.Variable(tf.random.uniform(shape=(1,), dtype=tf.float64)) for i in range(num_zeros)] # pure real
        self.ps = [tf.Variable(tf.random.uniform(shape=(2,), dtype=tf.float64)) for i in range(num_poles)] # complex (includ CC)
        self.g = tf.Variable(tf.random.uniform(shape=(1,), dtype=tf.float64)) # gain
        self.train_vars = self.zs + self.ps + [self.g]
        self.learning_rate = learning_rate

        for p in self.ps:
            p.assign(tf.clip_by_norm(p, MAX_POLE_NORM))

    def freqz(self, worN, fs):
        """
        Returns an audioSample of the frequency response corresponding to 
        the zpk filter. Inputs are the same as scipy.signal.freqz_zpk

        ---------------------------------------------------------------------
        INPUTS
        ---------------------------------------------------------------------
        worN			| (int) length of the frequency response (assumes
                        | single-sided)
        ---------------------------------------------------------------------
        fs  			| (int) sampling frequency
        ---------------------------------------------------------------------
        
        ---------------------------------------------------------------------
        OUTPUTS
        ---------------------------------------------------------------------
        (audioSample) containing the frequency response (frequency domain)
        of the filter
        ---------------------------------------------------------------------

        """

        if isinstance(worN, float): 
            print("Float given instead of integer for length of freqz. This will be converted to an integer")
            worN=int(worN)

        f, h = sig.freqz_zpk(self.get_ps(answer_complex=True), self.get_zs(), \
            self.get_k(), worN=worN,fs=fs)

        print(f, h)

        return audioSample(h, type="f", Fs=fs) 

    def get_zs(self):
        """
        list of the zeros of the filter
        """
        return [z.numpy()[0] for z in self.zs]

    def get_ps(self,answer_complex=False):
        """
        Returns the poles of this filter with options for whether to get them
        as complex numbers or as numpy arrays of the with [real(p), imag(p)]. 
        Note that providing the poles in this format would give you two 
        poles, the one given and the complex conjugate. Will be given as 
        floats, not as tr.Variable instances

        ---------------------------------------------------------------------
        INPUTS
        ---------------------------------------------------------------------
        answer_complex			| (bool) wither to return the poles as 
                                | complex numbers
        ---------------------------------------------------------------------
    
        ---------------------------------------------------------------------
        OUTPUTS
        ---------------------------------------------------------------------
        (list) containing the poles in the specificed format
        ---------------------------------------------------------------------
        """

        ps = []
        
        for p in self.ps:
            if answer_complex:
                ps.append(p.numpy()[0] + 1j*p.numpy()[1])
                ps.append(p.numpy()[0] - 1j*p.numpy()[1])
            else:
                ps.append(tuple(p.numpy()))

        return ps

    def get_k(self):
        """
        the gain of a filter
        """
        return self.g.numpy()[0]


if __name__ == "__main__":
    
    zpk_filter = ZPKOptimizableFilter()


    freq_response = zpk_filter.freqz(512, 44e3)

    freq_response.plot(both=True)

    print(freq_response)

    
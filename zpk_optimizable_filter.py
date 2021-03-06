"""

Class representing a filter that can be learned by adjusting poles, 
zeros, and a gain value. It provides tools for building optimizers, 
and is meant to be used within or as a parent class of a class that 
contains the loss and training functions


@author: Tony Terrasa
based off of work and guidance of David Ramsay
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal as sig
from pythonAudioMeasurements.audioSample import audioSample
from pyAudioFilter.filter_helpers import plot_zpk


#plot_sos(target, FS)

MAX_POLE_NORM = 0.900
pi = np.pi

class ZPKOptimizableFilter:

    def __init__(self, num_zeros=2, num_poles=1, learning_rate=0.1):
        """

        Filter in zpk format for learning with TensorFlow. Defaults to a 
        biquad with two real zeros and pair of complex conjugate poles
        
        
        ---------------------------------------------------------------------
        INPUTS
        ---------------------------------------------------------------------
        num_zeros		| (int) the number of ~real~ zeros to include in this 
                        | filter
        ---------------------------------------------------------------------
        num_poles		| (int) number of PAIRS of ~complex-conjugate~ poles
        ---------------------------------------------------------------------
        learning_rate	| (float) can be used in gradient descent if used as 
                        | a parent class or if it's own learning rate is 
                        | desired
        ---------------------------------------------------------------------
        
        """

        self.zs = [tf.Variable(tf.random.uniform(shape=(1,), dtype=tf.float64)) for i in range(num_zeros)] # pure real
        self.ps = [tf.Variable(tf.random.uniform(shape=(2,), dtype=tf.float64)) for i in range(num_poles)] # complex (includ CC)
        self.g = tf.Variable(tf.random.uniform(shape=(1,), dtype=tf.float64)) # gain
        self.train_vars = self.zs + self.ps + [self.g]
        self.learning_rate = learning_rate

        for p in self.ps:
            p.assign(tf.clip_by_norm(p, MAX_POLE_NORM))

        # list addition with pure Python
        # all Tensorflow Variables
        self.train_vars = self.zs + self.ps + [self.g]

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

        w, h = sig.freqz_zpk(self.get_zs(), self.get_ps(answer_complex=True), \
            self.get_k(), worN=worN)

        return audioSample(h, type="f", Fs=fs) 

    
    def get_magnitudes_tf(self, freqs, fs=44.1e3):
        """

        Create a transfer function at the given frequencies for the zpk 
        as a tf object 

        Operates under the working principle of scipy.signal.freqz, that is:

                    jw                 -jw              -jwM
           jw    B(e  )    b[0] + b[1]e    + ... + b[M]e
        H(e  ) = ------ = -----------------------------------
                   jw                 -jw              -jwN
                A(e  )    a[0] + a[1]e    + ... + a[N]e

        source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz.html#scipy.signal.freqz
        
        ---------------------------------------------------------------------
        INPUTS
        ---------------------------------------------------------------------
        freqs   		| (tf.Tensor) list of frequencies at which to 
                        | calculate the gain  
        ---------------------------------------------------------------------
        
        
        ---------------------------------------------------------------------
        OUTPUTS
        ---------------------------------------------------------------------
        (tf.Tensor) containg the transfer function gains at the given 
        frequencies. Note that this is the 
        ---------------------------------------------------------------------
        
        """
        
        # complex valued, rad/samp omegas 
        omegas = 2*pi*freqs/fs
        jomegas = tf.complex(tf.zeros(tf.shape(freqs),dtype=freqs.dtype), 2*pi*freqs)
        ejw = tf.exp(jomegas) 


        # calculating the magnitude of the output using the distances to Poles
        # and zeroes
        numerator = 1
        denominator = 1

        for z in self.zs:
            complex_z = tf.complex(z, tf.cast(0., z.dtype)) 
            numerator *= (ejw - complex_z)
        for p in self.ps:
            complex_p1 = tf.complex(p[0], p[1])
            complex_p2 = tf.complex(p[0], -p[1])
            denominator *= (ejw - complex_p1) * (ejw - complex_p2)


        # have to make g a complex number for this calculation
        complex_g = tf.complex(self.g, tf.cast(0, self.g.dtype))

        # magnitude in linear units and db
        H = complex_g * (numerator / denominator)

        return H

    def clip_poles_by_norm(self, norm=1.0):
        """
        Clip the poles of this fliter by their norm IN PLACE. Typically done 
        to maintain stability through the optimization process.         

        
        ---------------------------------------------------------------------
        INPUTS
        ---------------------------------------------------------------------
        norm			| (int, float) maximum norm of a pole
        ---------------------------------------------------------------------
        
        """

        for p in self.ps:
            p.assign(tf.clip_by_norm(p, norm))


    def visualize(self, fs=44.1e3):
        """
        use filter_helpers.plot_zpk to vizualize the poles and
        zeros for this reponse
        """
        plot_zpk(self.get_zs(), self.get_ps(answer_complex=True), self.get_k())


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

    
    def get_train_vars(self): return self.train_vars


if __name__ == "__main__":
    
    zpk_filter = ZPKOptimizableFilter()

    a = audioSample(np.arange(10000))
    freqs = a.f()

    tf_freqs = tf.constant(freqs, dtype=tf.float64)

    print("frequencies", tf_freqs)

    h = zpk_filter.get_magnitudes_tf(tf_freqs)

    mags = tf.abs(h) # this is now a float64
    phases = tf.math.angle(h) # also float64

    print("h_mags equals:", mags)
    print("h_phase equals:", phases)

    


    # freq_response = zpk_filter.freqz(512, 44e3) 
    # freq_response.plot(both=True)
    # print(freq_response)

    
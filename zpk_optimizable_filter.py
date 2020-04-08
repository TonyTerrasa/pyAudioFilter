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
from filter_helpers import *


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


        print(self.train_vars)


    def get_zs(self):
        return [z.numpy()[0] for z in self.zs]

    def get_ps(self,answer_complex=False):
        ps = []
        
        for p in self.ps:
            if answer_complex:
                ps.append(p.numpy()[0] + 1j*p.numpy()[1])
                ps.append(p.numpy()[0] - 1j*p.numpy()[1])
            else:
                ps.append(p.numpy())

        return ps


if __name__ == "__main__":
    pass
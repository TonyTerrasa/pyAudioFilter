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
from pyAudioFilter.filter_helpers import *
from pyAudioFilter.zpk_optimizable_filter import *


#plot_sos(target, FS)

MAX_POLE_NORM = 0.900

class BiQuadOptimizer(ZPKOptimizableFilter):

    def __init__(self, learning_rate=0.1):
        # optimizing/pushing the poles and zeros of this system
        super().__init__(num_zeros=2,num_poles=1,learning_rate=learning_rate)

    # for some reason, this breaks without the tf.function decorator
    #@tf.function
    def loss(self, w, mag_target):
        # cartesian coordinates of the input signals
        # note that we assume a norm of 1, that is a pure sinosoid
        x_loc  = tf.cos(w)
        y_loc  = tf.sin(w)

        #print(x_loc)
        #print(x_loc, y_loc)

        # calculating the magnitude of the output using the distances to Poles
        # and zeroes
        numerator = 1
        denominator = 1

        for z in self.zs:
            numerator *= tf.sqrt(tf.square(x_loc - z[0]) + tf.square(y_loc - 0) )
        for p in self.ps:
            denominator *= tf.sqrt(tf.square(x_loc - p[0])  +  tf.square(y_loc - p[1]))


        # numerator and denominator of the transfer function for these frequencies
        # are product of ht edistances to the zeroes and poles
        # numerator = dist_z1 * dist_z2
        # denominator = dist_p1 * dist_p2
        #print(denominator)

        # magnitude in linear units and db
        linear_magnitude = self.g * (numerator / denominator)


        #return linear_magnitude

        # note that tf.math.log is the natural log the log base 10 can be callculated
        # as log(x)/log(10)
        db_magnitude = 20 * tf.math.log(linear_magnitude) / tf.dtypes.cast(tf.math.log(10.0), dtype=tf.float64)

        return tf.reduce_sum(tf.square(db_magnitude - mag_target))/1000


    def train(self, w, mag_target):


        #print(self.current_loss)
            
        # calculate the gradients for each variable
        with tf.GradientTape() as t:

            self.current_loss = self.loss(w, mag_target)
            #print(self.train_vars)
            grads = t.gradient(self.current_loss, self.train_vars)

        # apply the gradients to the variables
        for i, var in enumerate(self.train_vars):
            var.assign_sub(self.learning_rate*grads[i])

        #print(grads)

        #print(tf.clip_by_norm(self.p, MAX_POLE_NORM))
        for p in self.ps:
            p.assign(tf.clip_by_norm(p, MAX_POLE_NORM))


if __name__ == "__main__":

    FS = 44100
    TARGET_FC = 2000 # cutoff frequency
    TARGET_ORDER = 2
    TARGET_TYPE = 'lowpass' # 'highpass', 'bandpass', 'bandstop'

    # target polar response
    # butterworth filter
    # this is returned coefficients (sos)
    target = sig.butter(TARGET_ORDER, TARGET_FC, TARGET_TYPE, analog=False, fs=FS, output='sos')


    #calc a target at some set of frequencies
    # target that we are using is the magntide in db
    # of the desired
    targ_freqs = np.array([100,300,2200,5000,10000,12500,15000,20000])
    w, mag_target = get_mag_targ(target, targ_freqs, fs=FS)


    NUM_EPOCHS = 1100
    optimizer = BiQuadOptimizer(learning_rate=0.001)

    losses = []
    zs = []
    ps = []

    for epoch in range(NUM_EPOCHS):

        optimizer.train(w, mag_target)
        losses.append(optimizer.current_loss)
        zs.append(optimizer.get_zs())
        ps.append(optimizer.get_ps()[0])
        print("loss: %f, z1: %f, z2: %f, p: %f,%f, g: %f"%(optimizer.current_loss, zs[-1][0], zs[-1][1], ps[-1][0], ps[-1][1], optimizer.g.numpy()[0]))
        print('-'*25)


    # ---------------------------------------------
    # plot the results
    # ---------------------------------------------

    # get the final poles and zeros
    z = optimizer.get_zs()
    p = optimizer.get_ps(answer_complex=True)

    # plot the target freq response
    plot_sos(target, FS, False)
    # plot the resulting freq response
    plot_zpk(z, p, optimizer.g.numpy()[0], FS, False)

    plt.show()

    # 
    # print(ps)
    # z1,z2 = zip(*zs)
    # p_r,p_i = zip(*ps)

    # plt.figure()
    # plt.plot(losses)
    # plt.plot(z1)
    # plt.plot(z2)
    # plt.plot(p_r)
    # plt.plot(p_i)
    # plt.legend(["loss", "z1", "z2", "p_{real}", "p_{image}"])
    # plt.xlabel("epochs")
    # plt.ylabel("loss")
    # plt.show()

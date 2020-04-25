"""
This defines a class that works with a data format already developed for handling
polar data and optimizing a 


@author: Tony Terrasa
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal as sig
import pickle 
from pythonAudioMeasurements.polarData import polarData
from pythonAudioMeasurements.microphone import Microphone
from pythonAudioMeasurements.microphoneArray import MicrophoneArray
from pyAudioFilter.zpk_optimizable_filter import ZPKOptimizableFilter


MAX_POLE_NORM = 0.9

class PolarOptimizer:


    def __init__(self, angles, freqs, microphones, learning_rate=0.1):
        """
        target_response should be iterable and indexable as 
        [angle, i] to yield an amplitude for target_freqs[i] in db

        microphone should be a polarData object
        """
        self.angles = angles
        self.freqs = freqs
        self.microphones = microphones

        # one filter per mic
        self.filters = [ZPKOptimizableFilter() for m in microphones]

        # pure Python list
        self.train_vars = []
        for filt in self.filters:
            self.train_vars.extend(filt.get_train_vars())

    @staticmethod
    def get_band_slices(vector, bounds):
        """
        Returns the lower bound index and the upper bound index for where
        bounds[0] < vector < bounds[1]. 
        
        
        ---------------------------------------------------------------------
        INPUTS
        ---------------------------------------------------------------------
        vector			| (iterable-1d) must be sorted
        ---------------------------------------------------------------------
        bounds			| (iterable) of size (2,). The 0 index is the 
                        | lower bound to be found in vector and the 1 index 
                        | is the upper bound
        ---------------------------------------------------------------------
        
        
        ---------------------------------------------------------------------
        OUTPUTS
        ---------------------------------------------------------------------
        lower_bound		| (int) index of the first element in vector to 
                        | satisfy the condition vector[lower_bound]>bounds[0]
        ---------------------------------------------------------------------
        upper_bound		| (int) index of the last element in vector to 
                        | satisfy the condition vector[upper_bound]<bounds[1]
        ---------------------------------------------------------------------
        
        """

        lower_bound = tf.reduce_min(tf.where(vector > bounds[0]))
        upper_bound = tf.reduce_max(tf.where(vector < bounds[1]))

        return lower_bound, upper_bound

    def get_freq_band(self, bounds):
        """
        See get_band_slices(self, vector, bounds) where vector=self.freqs
        """
        return PolarOptimizer.get_band_slices(self.freqs, bounds)

    def get_angle_band(self, bounds):
        """
        See get_band_slices(self, vector, bounds) where vector=self.angles
        """
        return PolarOptimizer.get_band_slices(self.angles, bounds)

    def loss(self, stop_band_theta, target_freq_range, threshold=-20.):
        """

        Calculates the loss aiming for a stop band between two angles and 
        and for a target frequency range

        ---------------------------------------------------------------------
        stop_band_theta		| (2-entry int,float) containing the coordinates
        ---------------------------------------------------------------------
        target_freq_range	| (2-entry int,flot) containing upper and lower
                            | bounds of the frequency range over which to 
                            | caluculate the loss
        ---------------------------------------------------------------------
        threshold			| (float) threshold in db to be considered 
                            | "stopped" (effective max magnitude)
        ---------------------------------------------------------------------

        """

        system_response = self.get_system_resonse()
        # get the magnitudes, emulating log10
        mags = tf.abs(system_response)
        system_response_db = 20*tf.log(mags)/tf.log(tf.constant(10, dtype=mags.dtype))

        # get responses in our angle and frequency range
        theta_lower, theta_upper = self.get_angle_band(stop_band_theta)
        freq_lower, freq_upper = self.get_freq_band(stop_band_theta)

        region_of_interest = system_response_db[theta_lower:theta_upper, freq_lower:freq_upper]

        # zero out anything below the threshold 
        loss = 0

        # loop through all magnitudes of the 
        for mag in tf.reshape(region_of_interest, (tf.size(region_of_interest))):
            # add how much greate than the threshold
            if mag > threshold:
                loss += mag-threshold

        return loss
            
    def train(self, stop_band_theta, target_freq_range, threshold=-20.):


        #print(self.current_loss)
            
        # calculate the gradients for each variable
        with tf.GradientTape() as t:

            self.current_loss = self.loss(stop_band_theta, target_freq_range, threshold=-20.)
            grads = t.gradient(self.current_loss, self.train_vars)

        # apply the gradients to the variables
        for i, var in enumerate(self.train_vars):
            var.assign_sub(self.learning_rate*grads[i])

        #print(grads)

        for p in self.ps:
            p.assign(tf.clip_by_norm(p, MAX_POLE_NORM))

    def response_length(self):
        """
        Returns the length of the responses stored in this instance
        """

        return len(self.microphones[0][1,:])


    def get_system_resonse(self):
        """
        Get the 2-d array representing the total system response including 
        the filters
        
        ---------------------------------------------------------------------
        OUTPUTS
        ---------------------------------------------------------------------
        response		| (numpy.array-2d) (anglesxresponse_length) assumes
                        | frequenc domain
        ---------------------------------------------------------------------
        
        """

        system_resonse = tf.zeros(self.microphones[0].shape, dtype=tf.complex128)
        tile_shape = tf.constant([len(self.angles), 1]) 

        for mic, filt in zip(self.microphones, self.filters):

            filt_response = filt.get_magnitudes_tf(self.freqs)

            # of the same shape as each microphone response. Contains a copy of 
            # filt_response for each microphone
            # reshape requred to use tf.tile
            filt_matrix = tf.tile(tf.reshape(filt_response,[1, len(filt_response)]), tile_shape)

            # apply filter in the frequency domain
            # total response is addition of all microphones
            system_resonse += mic*filt_matrix


        return system_resonse





if __name__ == "__main__":
    filename = "/home/terrasa/UROP/polar-measurement/data/19_Jan15_fixedpkls/spv1840.pkl" 

    pd = polarData.fromPkl(filename)

    mic_1 = Microphone(pd, (0,0))
    mic_2 = Microphone(pd, (100,200))
    mic_3 = Microphone(pd, (-100,200))
    mic_array =  MicrophoneArray([mic_1, mic_2, mic_3])
    # mic_array.visualize()


    angles, freqs, mic_responses = mic_array.tf_prep()

    angles = tf.constant(angles)
    freqs = tf.constant(freqs)
    mic_responses = [tf.constant(mic) for mic in mic_responses]

    po = PolarOptimizer(angles, freqs, mic_responses)

    response = po.get_system_resonse()

    print("Total Reponse:", response)








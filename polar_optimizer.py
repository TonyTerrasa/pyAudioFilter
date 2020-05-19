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


    def __init__(self, angles, freqs, microphones, fs=44.1e3, learning_rate=0.1):
        """
        target_response should be iterable and indexable as 
        [angle, i] to yield an amplitude for target_freqs[i] in db

        microphone should be a polarData object
        """
        self.angles = angles
        self.freqs = freqs
        self.microphones = microphones
        self.fs = fs
        self.learning_rate = learning_rate

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

        lower_bound = tf.reduce_min(tf.where(vector >= bounds[0]))
        upper_bound = tf.reduce_max(tf.where(vector <= bounds[1]))

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

    def loss(self, stop_band_theta, target_freq_range, threshold=-30., \
        above_thresh=-10., k_below=0.001, k_above=0.001):
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

        # get the magnitudes, log10
        mags = tf.abs(system_response)
        system_response_db = 20*tf.math.log(mags)/tf.math.log(tf.constant(10, dtype=mags.dtype))

        # get responses in our angle and frequency range
        theta_lower, theta_upper = self.get_angle_band(stop_band_theta)
        freq_lower, freq_upper = self.get_freq_band(target_freq_range)

        # just the angle range over the frequency range
        region_of_interest = system_response_db[theta_lower:theta_upper, freq_lower:freq_upper]

        # get the indexes that are above the given threshold
        above_threshold_idx = tf.where(region_of_interest > threshold)

        # get the values of the ROI at these indeces
        above_threshold_vals = tf.gather_nd(region_of_interest, above_threshold_idx)

        # the loss is the sum of how far away each is from the threshold
        loss = k_below * tf.reduce_sum(above_threshold_vals-threshold)


        # now, calculate loss from squashing the same frequencies outside 
        # of the target theta range
        roi_below_theta_min = system_response_db[:theta_lower, freq_lower:freq_upper]
        roi_above_theta_max = system_response_db[theta_upper:, freq_lower:freq_upper]

        # where pass-band is below the threshold
        below_thresh_riobtm_idx = tf.where(roi_below_theta_min < above_thresh)
        below_thresh_rioatm_idx = tf.where(roi_above_theta_max < above_thresh)


        below_thresh_btm_vals = tf.gather_nd(roi_below_theta_min, below_thresh_riobtm_idx)
        below_thresh_atm_vals = tf.gather_nd(roi_above_theta_max, below_thresh_rioatm_idx)

        loss += k_above * tf.reduce_sum(threshold - below_thresh_btm_vals)
        loss += k_above * tf.reduce_sum(threshold - below_thresh_atm_vals)

        return loss
            
    def train(self, stop_bands_theta, target_freq_range, threshold=-20.):


        self.current_loss = 0
            
        # calculate the gradients of the loss for each variable
        with tf.GradientTape() as t:

            for stop_band in stop_bands_theta:
                self.current_loss += self.loss(stop_band, target_freq_range,\
                    threshold=threshold)

            grads = t.gradient(self.current_loss, self.train_vars)

        # apply the gradients to the variables
        for i, var in enumerate(self.train_vars):
            var.assign_sub(self.learning_rate*grads[i])


        for filt in self.filters:
            filt.clip_poles_by_norm(0.9)

    def response_length(self):
        """
        Returns the length of the responses stored in this instance
        """

        return len(self.microphones[0][1,:])


    def get_system_resonse(self):
        """
        Get the 2-d array representing the total system response including 
        the filters in the frequency domain
        
        ---------------------------------------------------------------------
        OUTPUTS
        ---------------------------------------------------------------------
        response		| (numpy.array-2d) (anglesxresponse_length) assumes
                        | frequenc domain
        ---------------------------------------------------------------------
        
        """

        # allocating
        system_resonse = tf.zeros(self.microphones[0].shape, dtype=tf.complex128)

        # necessary tiling shape for the frequency response of each filter
        # used to create a matrix where each row is identical and is the 
        # freq response of the filter of that microphone, allowing for element
        # wise multiplication to apply the filter
        tile_shape = tf.constant([len(self.angles), 1]) 

        for mic, filt in zip(self.microphones, self.filters):

            filt_response = filt.get_magnitudes_tf(self.freqs, fs=self.fs)

            # of the same shape as each microphone response. Contains a copy of 
            # filt_response for each microphone
            # reshape requred to use tf.tile
            filt_matrix = tf.tile(tf.reshape(filt_response,[1, len(filt_response)]), tile_shape)

            # apply filter in the frequency domain
            # total response is addition of all microphones
            system_resonse += mic*filt_matrix


        return system_resonse

    def to_polar_data(self):
        """
        Create a polarData object from this optimizer
        
        ---------------------------------------------------------------------
        OUTPUTS
        ---------------------------------------------------------------------
        pd				| (pythonAudioMeasurements.polarData) representing 
                        | this object in it's current state
        ---------------------------------------------------------------------
        
        """

        system_response = self.get_system_resonse()
        pd = polarData.from2dArray(system_response.numpy(), self.angles.numpy(), self.fs.numpy(), "f")

        return pd





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








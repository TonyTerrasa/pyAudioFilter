"""
This defines a class that works with a data format already developed for handling
polar data and optimizing a 


@author: Tony Terrasa
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.signal as sig
from filter_helpers import *
import pickle 
#from pythonAudioMeasurements.polarData import polarData


class PolarOptimizer(BiquadOptimizer):


    def __init__(self, mic_array, learning_rate=0.1):
        """
        target_response should be iterable and indexable as 
        [angle, i] to yield an amplitude for target_freqs[i] in db

        microphone should be a polarData object
        """
        super().__init__(learning_rate) 
        self.mic_array = mic_array



    def loss(self, stop_band_theta, target_freq_range):
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


        self.mic_array.








if __name__ == "__main__":
    print("hi there")

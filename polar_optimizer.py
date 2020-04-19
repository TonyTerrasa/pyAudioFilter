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
from pythonAudioMeasurements.polarData import polarData
from pythonAudioMeasurements.microphone import Microphone
from pythonAudioMeasurements.microphoneArray import MicrophoneArray
from pyAudioFilter.zpk_optimizable_filter import ZPKOptimizableFilter


class PolarOptimizer(ZPKOptimizableFilter):


    def __init__(self, angles, freqs, microphones, learning_rate=0.1):
        """
        target_response should be iterable and indexable as 
        [angle, i] to yield an amplitude for target_freqs[i] in db

        microphone should be a polarData object
        """
        super().__init__(learning_rate=learning_rate) 
        self.angles = angles
        self.freqs = freqs
        self.microphones = microphones



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

        pass










if __name__ == "__main__":
    filename = "/home/terrasa/UROP/polar-measurement/data/19_Jan15_fixedpkls/spv1840.pkl" 

    pd = polarData.fromPkl(filename)

    mic_1 = Microphone(pd, (0,0))
    mic_2 = Microphone(pd, (100,200))
    mic_3 = Microphone(pd, (-100,200))

    mic_array =  MicrophoneArray([mic_1, mic_2, mic_3])
    mic_array.visualize()


    angles, freqs, mic_1_numpy = mic_1.tf_prep()
    angles, freqs, mic_2_numpy = mic_2.tf_prep()
    angles, freqs, mic_3_numpy = mic_3.tf_prep()


    po = PolarOptimizer(angles, freqs, [mic_1_numpy, mic_2_numpy, mic_3_numpy])




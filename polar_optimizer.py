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


    def __init__(target_response, target_freqs, microphone_response, learning_rate=0.1):
        """
        target_response should be iterable and indexable as 
        [angle, i] to yield an amplitude for target_freqs[i] in db

        microphone should be a polarData object
        """
        super().__init__(learning_rate) 
        self.target_freqs = target_freqs
        self.target_response = target_response




if __name__ == "__main__":
    print("hi there")

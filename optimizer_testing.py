import tensorflow as tf
import numpy as np
from pythonAudioMeasurements.polarData import polarData
from pythonAudioMeasurements.microphone import Microphone
from pythonAudioMeasurements.microphoneArray import MicrophoneArray
from pyAudioFilter.polar_optimizer import PolarOptimizer



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



def test_bands():

    print(po.freqs)
    lower, upper = po.get_freq_band([100,3000])
    print("lower", lower, po.freqs[lower])
    print("upper", upper, po.freqs[upper])

    print(po.angles)
    lower, upper = po.get_angle_band([20, 180])
    print("lower", lower, po.freqs[lower])
    print("upper", upper, po.freqs[upper])



if __name__ == "__main__":

    test_bands()
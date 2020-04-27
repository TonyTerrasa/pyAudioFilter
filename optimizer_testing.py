import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pythonAudioMeasurements.polarData import polarData
from pythonAudioMeasurements.microphone import Microphone
from pythonAudioMeasurements.microphoneArray import MicrophoneArray
from pyAudioFilter.polar_optimizer import PolarOptimizer


tf.config.set_visible_devices([], 'GPU')


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

po = PolarOptimizer(angles, freqs, mic_responses, learning_rate=0.001)



def test_bands():

    print(po.freqs)
    lower, upper = po.get_freq_band([100,3000])
    print("lower", lower, po.freqs[lower])
    print("upper", upper, po.freqs[upper])

    print(po.angles)
    lower, upper = po.get_angle_band([20, 180])
    print("lower", lower, po.angles[lower])
    print("upper", upper, po.angles[upper])


def test_loss():
    loss = po.loss([10,60], [500,1000])
    print(loss)

def test_training_session():

    stop_band_theta = [10,60]
    target_range_freq = [500, 1000]

    losses = []

    for epoch in range(30):

        po.train(stop_band_theta, target_range_freq, threshold=-40.)
        losses.append(po.current_loss.numpy())

    print(losses)
    plt.plot(losses)
    plt.show()




if __name__ == "__main__":

    test_training_session()
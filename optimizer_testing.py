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

locations = [(0,0), (100,200), (-100, 200), (-100,-200), (100,-200)]

num_mics = 20
box_width = 600 # mm
locations = box_width*np.random.rand(num_mics,2)
mic_array =  MicrophoneArray([Microphone(pd, loc) for loc in locations])
mic_array.visualize()


angles, freqs, mic_responses = mic_array.tf_prep()

angles = tf.constant(angles)
freqs = tf.constant(freqs)
mic_responses = [tf.constant(mic) for mic in mic_responses]
fs = tf.constant(44.1e3, dtype=freqs.dtype)

po = PolarOptimizer(angles, freqs, mic_responses, fs=fs, learning_rate=0.01)



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

    stop_band_theta = np.array([45, 135])
    stop_bands_theta = np.array([stop_band_theta, stop_band_theta+180])

    print(stop_bands_theta)

    target_range_freq = [500, 1000]

    losses = []

    for epoch in range(60):

        po.train(stop_bands_theta, target_range_freq, threshold=-20.)
        losses.append(po.current_loss.numpy())

        if epoch % 10 == 0:
            pd = po.to_polar_data()
            pd.plotFreqs(np.linspace(target_range_freq[0], target_range_freq[1], 5),\
                title="epoch: %d, loss: %.2f"%(epoch, losses[-1]))

    print(losses)
    plt.figure(2)
    plt.plot(losses)
    plt.title("Loss over Epoch. Stop band: theta=(%d,%d), f=(%.2f, %.2f)"%(\
        10,60,500,1000))

    pd = po.to_polar_data()
    pd.plotFreqs(np.linspace(target_range_freq[0], target_range_freq[1], 5), show=False)

    plt.show()

def test_filter_visualize():

    po.filters[0].visualize()

def test_to_polar():
    pd = po.to_polar_data()
    pd.plotFreqs([100,1000])



if __name__ == "__main__":

    # test_filter_visualize()
    # test_to_polar()

    test_training_session()
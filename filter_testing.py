from __future__ import print_function
import numpy as np
import tensorflow as tf

from pythonAudioMeasurements.polarData import polarData
from pythonAudioMeasurements.audioSample import audioSample
from pyAudioFilter.zpk_optimizable_filter import ZPKOptimizableFilter



def test_get_vars():
    """
    Test getter functions
    """

    filt = ZPKOptimizableFilter(num_zeros=2,num_poles=1)

    print(filt.get_ps(answer_complex=False))
    print(filt.get_ps(answer_complex=True))
    print(filt.get_zs())
    print(filt.get_k())

def test_freqz():
    """
    Verify that the freqz function works
    """

    filt = ZPKOptimizableFilter(num_zeros=2,num_poles=1)

    freq_response = filt.freqz(10e3, 44e3)
    freq_response.plot(both=True)
    # print(freq_response.data)
    # print(freq_response.f())

def test_apply_filter():
    """
    Application of a zpk filter to a polarData object
    """

    
    filename = "/home/terrasa/UROP/polar-measurement/data/19_Jan15_fixedpkls/spv1840.pkl" 
    pd = polarData.fromPkl(filename)
    filt = ZPKOptimizableFilter(num_zeros=2,num_poles=1)
    worN = len(pd[0].f())
    fs = pd[0].fs
    filt_freqz = filt.freqz(worN, fs)

    filt_freqz.plot(both=True, fig=3, show=False, figtitle="Filter")

    pd.setType("f")

    test_frequencies = ([100,1000,1240, 10000])
    pd.plotFreqs(test_frequencies, fig=1, show=False, title="BEFORE")
    pd.applyFilter(filt_freqz)
    pd.plotFreqs(test_frequencies, fig=2, show=True, title="AFTER")



def pd_convert():
    """
    Application of a zpk filter to a polarData object
    """

    
    filename = "/home/terrasa/UROP/polar-measurement/data/19_Jan15_fixedpkls/spv1840.pkl" 
    pd = polarData.fromPkl(filename)

    print(pd.toTF().shape)


if __name__ == "__main__":
    # test_get_vars()
    # test_freqz()
    # test_apply_filter()
    pd_convert()
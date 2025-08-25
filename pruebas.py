import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm



V = 2.54
angulo = 180
angulo_60 = 120
V_60 = V*(angulo_60 / angulo)

V5 = 5
V5_bits = 1023
V_60_bits = (V5_bits * V_60) / V5
print("Voltaje a 60 grados:", V_60 )
print("Voltaje a 60 grados en bits:", V_60_bits)

"""def FourierTransform(signal):
    N = len(signal)
    transformed_signal = np.fft.fft(signal) / N
    return transformed_signal"""

def FourierTransform(t, y, freqs):
    N = len(t)
    T = t[1] - t[0]  # Assuming uniform spacing
    transformed_signal = np.array([np.sum(y * np.exp(-2j * np.pi * f * t)) for f in freqs]) * T / N
    return transformed_signal

def InverseFourierTransform(t, freqs, transformed_signal):
    N = len(t)
    T = t[1] - t[0]  # Assuming uniform spacing
    reconstructed_signal = np.array([np.sum(transformed_signal * np.exp(2j * np.pi * f * t)) for t in t]) * N / T
    return reconstructed_signal


def GaussianFilter(freqs, center_freq, bandwidth):
    return norm.pdf(freqs, loc=center_freq, scale=bandwidth)

def generate_data(tmax,dt,A,freq,noise):
    '''Generates a sin wave of the given amplitude (A) and frequency (freq),
    sampled at times going from t=0 to t=tmax, taking data each dt units of time.
    A random number with the given standard deviation (noise) is added to each data point.
    ----------------
    Returns an array with the times and the measurements of the signal. '''
    ts = np.arange(0,tmax+dt,dt)
    return ts, np.random.normal(loc=A*np.sin(2*np.pi*ts*freq),scale=noise)
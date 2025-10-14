import numpy as np
import matplotlib.pyplot as plt
import itertools

#=======TASK 3 - DISCRETE FOURIER TRANSFORM=======

def manual_dft(signal, sampling_frequency=1.0, show_plot=True):
    """
    Compute the Discrete Fourier Transform (DFT) manually (magnitude only).
    """
    N = len(signal)
    magnitudes = np.zeros(N)

    # Compute real and imaginary parts for each frequency bin
    for k in range(N):
        real_part = 0.0  
        imag_part = 0.0   
        for n in range(N):
            angle = (2 * np.pi * k * n) / N
            real_part += signal[n] * np.cos(angle)
            imag_part -= signal[n] * np.sin(angle)

        real_part /= N
        imag_part /= N
        magnitudes[k] = np.sqrt(real_part**2 + imag_part**2)

    # Rescale x-axis into frequency for visualization
    freqs = np.arange(N) * (sampling_frequency / N)

    if show_plot:
        half = N // 2
        plt.figure(figsize=(10,5))
        plt.stem(freqs[:half], magnitudes[:half], basefmt=" ", markerfmt="o")
        plt.title("Manual DFT Magnitude Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.show()

    return freqs, magnitudes


def numpy_dft(signal, sampling_frequency=1.0, show_plot=False):
    """
    Compute the DFT using NumPy's FFT implementation.
    """
    N = len(signal)
    F = np.fft.fft(signal) / N  
    freqs = np.fft.fftfreq(N, d=1/sampling_frequency)

    if show_plot:
        mask = freqs >= 0   
        plt.figure(figsize=(10,5))
        plt.stem(freqs[mask], np.abs(F[mask]), basefmt=" ")
        plt.title("NumPy FFT Magnitude Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.show()

    return freqs, F


# Create a simple sine wave
fs = 1000               
t = np.linspace(0, 1, fs)     
f_signal = 50                 
signal = np.sin(2 * np.pi * f_signal * t)

# Compute DFT manually
freqs_manual, mags_manual = manual_dft(signal, sampling_frequency=fs, show_plot=True)

# Compute DFT using NumPy FFT
freqs_numpy, F_numpy = numpy_dft(signal, sampling_frequency=fs, show_plot=True)



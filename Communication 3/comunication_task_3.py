import numpy as np
import matplotlib.pyplot as plt

#=======TASK 2 - CARRIER SIGNAL MODULATION========

def amplitude_shift_keying(carrier_amplitude, 
                         dc_offset, 
                         modulation_amplitude, 
                         modulation_frequency, 
                         carrier_frequency, 
                         signal_duration=None, 
                         sampling_frequency=10000,
                         array_length=None,
                         create_plot=True,
                         show_message_signal=False):
    """
    Generate an amplitude modulated carrier signal.
    """
    # Determine number of samples
    if array_length is not None:
        N = array_length
    elif signal_duration is not None:
        N = int(sampling_frequency * signal_duration)
    else:
        raise ValueError("Either signal_duration or array_length must be provided.")

    # Build time, message, and carrier signals
    time_values = np.arange(N) / sampling_frequency
    message_signal = dc_offset + modulation_amplitude * np.sin(2 * np.pi * modulation_frequency * time_values)
    carrier_wave = carrier_amplitude * np.sin(2 * np.pi * carrier_frequency * time_values)
    modulated_signal = message_signal * carrier_wave

    if create_plot:
        # Plot AM waveform and envelopes
        plt.figure(figsize=(12, 6))
        plt.plot(time_values, modulated_signal, label="Amplitude Modulated Signal")
        plt.plot(time_values,  message_signal * carrier_amplitude, 'r--', alpha=0.7, label="Envelope (upper)")
        plt.plot(time_values, -message_signal * carrier_amplitude, 'r--', alpha=0.7, label="Envelope (lower)")

        # Optionally include original modulation signal
        if show_message_signal:
            scaled_message_signal = (message_signal - np.mean(message_signal)) * carrier_amplitude
            plt.plot(time_values, scaled_message_signal, 'g-', alpha=0.8, label="Original Modulation Signal (scaled)")

        plt.title(f"Amplitude Modulated Signal: Carrier = {carrier_frequency} Hz, Modulation = {modulation_frequency} Hz")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return time_values, modulated_signal

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


# Create a simple sine wave using Task 2

sampling_frequency = 1000

time_array, modulated_carrier_wave = amplitude_shift_keying(
    carrier_amplitude=1.0,
    dc_offset=1.0,
    modulation_amplitude=0.5,
    modulation_frequency=50,
    carrier_frequency=100,
    array_length=1000,
    sampling_frequency=sampling_frequency,
    show_message_signal=False,
    create_plot=False
)

# Compute DFT manually
freqs_manual, mags_manual = manual_dft(modulated_carrier_wave, sampling_frequency=sampling_frequency, show_plot=True)

# Compute DFT using NumPy FFT
freqs_numpy, F_numpy = numpy_dft(modulated_carrier_wave, sampling_frequency=1000, show_plot=True)



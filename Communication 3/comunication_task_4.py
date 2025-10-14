import numpy as np
import matplotlib.pyplot as plt
import itertools

#=======PREREQUISITES========

def amplitude_modulation(carrier_amplitude, 
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


#=======TASK 4 - BANDPASS FILTER APPLICATIONS (CONVOLUTION)=======

def bandpass_filter(signal_time_domain, signal, sampling_frequency, low_cutoff, high_cutoff, show_plots=True, add_noise=True, noise_strength=0.5):
    """
    Apply a square bandpass filter to a signal in the frequency domain.
    """
    N = len(signal)

    # Optionally add synthetic noise
    if add_noise:
        mains_noise = noise_strength * np.sin(2 * np.pi * 50 * signal_time_domain)
        high_freq_noise = noise_strength * np.sin(2 * np.pi * 500 * signal_time_domain)
        random_noise = noise_strength * np.random.randn(len(signal_time_domain))
        noisy_signal = signal + mains_noise + high_freq_noise + random_noise
    else:
        noisy_signal = signal

    # FFT, apply filter mask, then inverse FFT
    freqs = np.fft.fftfreq(N, 1/sampling_frequency)
    spectrum = np.fft.fft(noisy_signal)
    low_cutoff_array = np.abs(freqs) >= low_cutoff
    high_cutoff_array = np.abs(freqs) <= high_cutoff
    filter_mask = np.logical_and(low_cutoff_array, high_cutoff_array).astype(float)
    filtered_spectrum = spectrum * filter_mask
    filtered_signal_time_domain = np.fft.ifft(filtered_spectrum).real

    if show_plots:
        plt.figure(figsize=(14,12))

        # Plot original, noisy, and filtered signals
        plt.subplot(5,1,1)
        plt.plot(signal_time_domain, signal, label="Clean Signal")
        plt.title("Original Signal")
        plt.grid(True)
        plt.legend()

        plt.subplot(5,1,2)
        plt.plot(signal_time_domain, noisy_signal, label="Noisy Signal", color="orange")
        plt.title("Signal + Noise")
        plt.grid(True)
        plt.legend()

        plt.subplot(5,1,3)
        plt.plot(freqs[:N//2], np.abs(spectrum[:N//2]))
        plt.title("Noisy Spectrum")
        plt.grid(True)

        plt.subplot(5,1,4)
        plt.plot(freqs[:N//2], filter_mask[:N//2])
        plt.title("Bandpass Filter Mask")
        plt.grid(True)

        plt.subplot(5,1,5)
        plt.plot(signal_time_domain, filtered_signal_time_domain, label="Filtered Signal", color="green")
        plt.title("Filtered Signal")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    return filtered_signal_time_domain, freqs, filter_mask, noisy_signal


# Example usage
sampling_frequency = 1000
low_cutoff = 75
high_cutoff = 125

time_array, modulated_carrier_wave = amplitude_modulation(
    carrier_amplitude=1.0,
    dc_offset=1.0,
    modulation_amplitude=0.5,
    modulation_frequency=20,
    carrier_frequency=100,
    array_length=500,
    sampling_frequency=sampling_frequency,
    show_message_signal=True,
    create_plot=False
)

filtered_waveform, frequencies, bandpass_mask, noisy_signal = bandpass_filter(
    time_array, modulated_carrier_wave, sampling_frequency, low_cutoff, high_cutoff, show_plots=True, add_noise=True, noise_strength=0.5
)

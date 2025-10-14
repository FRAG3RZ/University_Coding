import numpy as np
import matplotlib.pyplot as plt
import itertools

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

time_array, modulated_carrier_wave = amplitude_shift_keying(
    carrier_amplitude=1.0,
    dc_offset=1.0,
    modulation_amplitude=0.5,
    modulation_frequency=20,
    carrier_frequency=100,
    array_length=500,
    sampling_frequency=1000,
    show_message_signal=False,
    create_plot=True
)

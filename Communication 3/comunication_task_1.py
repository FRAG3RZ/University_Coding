import numpy as np
import matplotlib.pyplot as plt
import itertools

#=======TASK 1 - SAMPLING FREQUENCIES========

def compare_sampling(amplitude, frequency, num_samples_list, sample_frequencies_list, show_original_frequency=False):
    """
    Compare discrete sine wave sampling for combinations of sample counts and sampling frequencies.
    """
    combos = list(itertools.product(num_samples_list, sample_frequencies_list))

    _, axes = plt.subplots(len(combos), 1, figsize=(14, 3*len(combos)))

    end_value_array = []

    if len(combos) == 1:
        axes = [axes]

    for ax, (num_samples, fs) in zip(axes, combos):
        # Generate sampled sine wave
        t = np.arange(num_samples) / fs
        wave = amplitude * np.sin(2 * np.pi * frequency * t)
        end_value_array.append(wave)

        # Optionally show continuous (dense) sine wave
        if show_original_frequency:
            t_dense = np.linspace(0, num_samples/fs, 1000)
            wave_dense = amplitude * np.sin(2 * np.pi * frequency * t_dense)
            ax.plot(t_dense, wave_dense, 'r--', alpha=0.6, label="Original continuous")

        # Plot discrete samples
        ax.plot(t, wave, marker='o', label=f"N={num_samples}, fs={fs}Hz")
        ax.set_title(f"Sine Wave: f={frequency}Hz, N={num_samples}, fs={fs}Hz")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        ax.set_ylim(-amplitude*1.1, amplitude*1.1)
        ax.grid(True)
        ax.legend()

    plt.tight_layout(pad=3.0)
    plt.show()

    return end_value_array


values = compare_sampling(
    amplitude=1.0,
    frequency=10,
    num_samples_list=[50],
    sample_frequencies_list=[11, 20, 21, 50],
    show_original_frequency=True
)

print(values)

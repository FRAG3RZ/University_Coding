import numpy as np
import matplotlib.pyplot as plt
from InquirerPy import inquirer
from rich.console import Console
from rich.table import Table
import time
from rich import box
from time import sleep
from rich import print as rprint

# =======Utility Funtions=======

def string_to_hex_array(word):
    # Make a list of each character and then append one by one
    word = list(f"{word}")
    hex_final = []
    for char in word:
        hex_final.append(hex(ord(char)))
    return hex_final


def hex_to_binary_raw(word):
    bit_string = "".join(f"{int(h, 16):08b}" for h in word)
    return bit_string


def binary_to_hex(binary_string):
    # Pad to multiple of 8 bits if needed
    padded = binary_string + "0" * ((8 - len(binary_string) % 8) % 8)

    # Group into bytes and convert each to hex
    hex_list = [hex(int(padded[i : i + 8], 2)) for i in range(0, len(padded), 8)]

    return hex_list


def hex_to_string(hex_list):
    chars = [chr(int(h, 16)) for h in hex_list]
    return "".join(chars)


def display_results_table(results):
    console = Console()
    table = Table(
        title="📊 PARAMETER SWEEP RESULTS",
        title_style="bold cyan",
        header_style="bold magenta",
        box=box.SQUARE,  # adds clean horizontal/vertical lines between rows
        show_lines=True,  # adds lines between rows
    )

    # === Define table columns ===
    table.add_column("Baud Rate", justify="right")
    table.add_column("Sampling Rate", justify="right")
    table.add_column("Carrier Freq", justify="right")
    table.add_column("FSK Δf", justify="right")
    table.add_column("SNR (dB)", justify="right")
    table.add_column("Raw BER", justify="right")
    table.add_column("Retransmissions", justify="right")
    table.add_column("Failed Bytes", justify="right")
    table.add_column("Runtime", justify="right")

    # === Conditional coloring logic ===
    for row in results:
        baud, sample, carrier, FSK_dev, snr, raw_ber, retrans, failed_bytes, runtime = (
            row
        )

        # Color BER based on error severity
        ber_val = float(raw_ber)
        if ber_val > 0.2:
            ber_color = "[bold red]"
        elif ber_val > 0.05:
            ber_color = "[yellow]"
        else:
            ber_color = "[green]"

        # Color retransmissions (more retrans = worse)
        retrans_color = (
            "[green]"
            if int(retrans) == 0
            else ("[yellow]" if int(retrans) < 10 else "[red]")
        )

        # Color failed bytes
        fail_color = (
            "[green]"
            if int(failed_bytes) == 0
            else ("[yellow]" if int(failed_bytes) < 5 else "[red]")
        )

        table.add_row(
            str(baud),
            str(sample),
            str(carrier),
            str(FSK_dev),
            str(snr),
            f"{ber_color}{raw_ber}[/]",
            f"{retrans_color}{retrans}[/]",
            f"{fail_color}{failed_bytes}[/]",
            f"[cyan]{runtime}[/]",
        )

    console.print("\n")
    console.print(table)
    console.print("\n")

def _parse_number_list(s, cast=float):
    """Parse a comma-separated list into numbers with whitespace tolerance."""
    return [cast(x.strip()) for x in s.split(",") if x.strip() != ""]

def ask_simulation_params(mode="single"):
    """
    Collect interactive parameters for either a single run or a sweep.
    Returns a dict with a consistent schema.

    For mode='single':
      - Scalars: baud_rate, sampling_rate, carrier_frequency, SNR
    For mode='sweep':
      - Lists:   baud_rates, sampling_rates, carrier_frequencies, SNR_values
    In both:
      - error_method, fading_model, retransmit_enabled, auto_adjust, message
    """
    assert mode in ("single", "sweep")

    mode_choice = "Frequency Shift Keying (FSK)"  # constant in your code

    if mode == "single":
        baud_rate = int(
            inquirer.text(
                message="Enter Baud Rate:",
                default="50",
                validate=lambda val: val.isdigit() and int(val) > 0,
            ).execute()
        )

        sampling_rate = int(
            inquirer.text(
                message="Enter Sampling Rate:",
                default="100000",
                validate=lambda val: val.isdigit() and int(val) > 0,
            ).execute()
        )

        carrier_frequency = int(
            inquirer.text(
                message="Enter Carrier Frequency (Hz):",
                default="1000",
                validate=lambda val: val.isdigit() and int(val) > 0,
            ).execute()
        )

        SNR = float(
            inquirer.text(
                message="Enter desired SNR (dB):",
                default="18",
                validate=lambda val: val.replace(".", "", 1).isdigit() and float(val) > 0,
            ).execute()
        )
    else:
        baud_rates = _parse_number_list(
            inquirer.text(
                message="Enter Baud Rates (comma separated):",
                default="75,100,125",
            ).execute(),
            cast=int,
        )

        sampling_rates = _parse_number_list(
            inquirer.text(
                message="Enter Sampling Rates (comma separated):",
                default="20000",
            ).execute(),
            cast=int,
        )

        carrier_frequencies = _parse_number_list(
            inquirer.text(
                message="Enter Carrier Frequencies (comma separated):",
                default="1000",
            ).execute(),
            cast=int,
        )

        SNR_values = _parse_number_list(
            inquirer.text(
                message="Enter SNR values (dB, comma separated):",
                default="6,12,18,24",
            ).execute(),
            cast=float,
        )

    error_method = inquirer.select(
        message="Select error checking/correction method:",
        choices=["None", "Parity Bit (even)", "Hamming (7,4)"],
        default="Parity Bit (even)",
    ).execute()

    fading_model = inquirer.select(
        message="Select channel fading model:",
        choices=["None", "Rayleigh", "Rician"],
        default="None",
    ).execute()

    retransmit_enabled = inquirer.confirm(
        message=("Enable retransmission for corrupted bytes?"
                 if mode == "single"
                 else "Enable retransmission for corrupted bytes during sweep?"),
        default=True,
    ).execute()

    auto_adjust = inquirer.confirm(
        message=("Allow automatic adjustment of sampling/carrier frequencies?"
                 if mode == "single"
                 else "Allow automatic parameter adjustment to ensure maximum stability during sweep?"),
        default=True,
    ).execute()

    message = inquirer.text(
        message="Enter message to transmit:", default="Hello World!"
    ).execute()

    # Assemble output
    params = {
        "mode_choice": mode_choice,
        "error_method": error_method,
        "fading_model": fading_model,
        "retransmit_enabled": retransmit_enabled,
        "auto_adjust": auto_adjust,
        "message": message,
    }

    if mode == "single":
        params.update(
            dict(
                baud_rate=baud_rate,
                sampling_rate=sampling_rate,
                carrier_frequency=carrier_frequency,
                SNR=SNR,
            )
        )
    else:
        params.update(
            dict(
                baud_rates=baud_rates,
                sampling_rates=sampling_rates,
                carrier_frequencies=carrier_frequencies,
                SNR_values=SNR_values,
            )
        )

    return params


def check_and_adjust_sampling(
    baud_rate,
    sampling_rate,
    FSK_deviation,
    carrier_frequency,
    sweep_mode=False,
    bw_fraction=0.20,
    gamma=0.75,
):
    """
    Verify & auto-adjust Fs, Δf, and (optionally) Fc to satisfy:
      • ≥20 samples/bit
      • ≥10 cycles/bit
      • Δf big enough (target γ·Rb) but ≤ BW cap
      • BW ≈ 2(Δf + Rb) ≤ bw_fraction · Fc
    Returns (sampling_rate, FSK_deviation, carrier_frequency)
    """
    Rb = float(baud_rate)
    fc = float(carrier_frequency)

    # Thresholds
    min_samples_per_bit = 20
    min_cycles_per_bit = 10

    # Derived
    samples_per_bit = sampling_rate / Rb
    cycles_per_bit = fc / Rb

    # Carrier needed to allow Δf_target = γ·Rb under BW cap
    fc_min_for_target = (
        2.0 * ((gamma + 1.0) * Rb)
    ) / bw_fraction  # = 10(γ+1)Rb for bw_fraction=0.2

    reasons = []
    if samples_per_bit < min_samples_per_bit:
        reasons.append(f"{samples_per_bit:.1f} samples/bit (<{min_samples_per_bit})")
    if cycles_per_bit < min_cycles_per_bit:
        reasons.append(f"{cycles_per_bit:.1f} cycles/bit (<{min_cycles_per_bit})")
    if fc < fc_min_for_target:
        reasons.append(
            f"Fc too low for BW cap at target Δf (Fc<{fc_min_for_target:.0f} Hz)"
        )

    # Compute BW cap at current (possibly updated) Fc
    bw_cap = bw_fraction * max(fc, fc_min_for_target)
    delta_f_max_allowed = bw_cap / 2.0 - Rb  # from 2(Δf+Rb) ≤ bw_cap

    # Choose Δf: as large as possible but ≤ cap, and ≥ gamma*Rb, and ideally ≥ 5% Fc
    delta_f_target = gamma * Rb
    # If Fc is very low, 5%·Fc can be smaller than target; ensure at least 1 Hz
    delta_f_min_practical = max(1.0, 0.05 * max(fc, fc_min_for_target))

    # If cap is negative, no Δf possible → must raise Fc or fail
    if delta_f_max_allowed <= 0:
        reasons.append("No deviation allowed by BW cap (Δf_max≤0)")

    warn = len(reasons) > 0

    if warn:
        print("\n⚠️  Warning: parameters may cause demodulation or bandwidth issues:")
        for r in reasons:
            print("   •", r)

        if sweep_mode:
            print("⚙️  Auto-adjusting for sweep mode...")

            # Auto-raise Fc if needed to allow target Δf under BW cap
            if fc < fc_min_for_target:
                fc = int(np.ceil(fc_min_for_target))

            # Recompute BW cap & Δf max with updated Fc
            bw_cap = bw_fraction * fc
            delta_f_max_allowed = bw_cap / 2.0 - Rb

            if delta_f_max_allowed <= 0:
                # Still infeasible: return current Fs/Δf=1 and keep Fc; caller may skip/log
                print(
                    "   → Still infeasible after Fc raise; keeping Δf=1 Hz (expect poor BER)."
                )
                return sampling_rate, 1, int(fc)

            # Δf choice
            delta_f = min(
                max(delta_f_target, delta_f_min_practical), delta_f_max_allowed
            )
            # Fs choice for samples/bit
            sampling_rate = int(max(sampling_rate, Rb * min_samples_per_bit))

            print(f"   → Sampling Rate set to {sampling_rate}")
            print(f"   → Carrier Frequency set to {int(fc)}")
            print(f"   → FSK Deviation set to {int(delta_f)}\n")
            return sampling_rate, int(delta_f), int(fc)

        # Interactive mode: propose adjustments
        auto_fix = inquirer.confirm(
            message="Auto-adjust Fs/Δf/Fc for reliable operation and BW cap?",
            default=True,
        ).execute()

        if auto_fix:
            if fc < fc_min_for_target:
                fc = int(np.ceil(fc_min_for_target))
            bw_cap = bw_fraction * fc
            delta_f_max_allowed = bw_cap / 2.0 - Rb
            if delta_f_max_allowed <= 0:
                print(
                    "❌ Even after raising Fc, BW cap allows no Δf. Lower baud or relax BW cap."
                )
                return sampling_rate, 1, int(fc)

            delta_f = min(
                max(delta_f_target, delta_f_min_practical), delta_f_max_allowed
            )
            sampling_rate = int(max(sampling_rate, Rb * min_samples_per_bit))

            print("\n🔧 Adjusting parameters:")
            print(f"   - Sampling Rate: → {sampling_rate}")
            print(f"   - Carrier Freq : → {int(fc)}")
            print(f"   - FSK Deviation: → {int(delta_f)}\n")
            return sampling_rate, int(delta_f), int(fc)
        else:
            print("⚠️ Proceeding without changes. Expect degraded performance.\n")

    # No warnings: still consider mild Δf improvement within cap
    bw_cap = bw_fraction * fc
    delta_f_max_allowed = bw_cap / 2.0 - Rb
    delta_f = min(
        max(FSK_deviation, delta_f_target, delta_f_min_practical),
        max(1.0, delta_f_max_allowed),
    )
    return sampling_rate, int(delta_f), int(fc)


# ==========================================================
# =============== ERROR DETECTION / CORRECTION =============
# ==========================================================


def add_parity_bit(byte_string, even=True):
    """Add a single parity bit to an 8-bit byte string."""
    ones = byte_string.count("1")
    parity = "0" if (ones % 2 == 0) == even else "1"
    return byte_string + parity


def check_parity_bit(block, even=True):
    """Check a single 9-bit block for parity errors and return 8-bit data + error flag."""
    if len(block) < 9:
        return block[:-1], 1  # invalid frame, count as error
    data, parity_bit = block[:8], block[-1]
    ones = data.count("1")
    expected = "0" if (ones % 2 == 0) == even else "1"
    error = int(parity_bit != expected)
    return data, error


def hamming74_encode(bit_string):
    encoded = ""
    for i in range(0, len(bit_string), 4):
        d = bit_string[i : i + 4].ljust(4, "0")
        d1, d2, d3, d4 = map(int, d)
        p1 = (d1 + d2 + d4) % 2
        p2 = (d1 + d3 + d4) % 2
        p3 = (d2 + d3 + d4) % 2
        encoded += f"{p1}{p2}{d1}{p3}{d2}{d3}{d4}"
    return encoded


def hamming74_decode(encoded_bits):
    decoded = ""
    corrected = 0
    for i in range(0, len(encoded_bits), 7):
        block = encoded_bits[i : i + 7]
        if len(block) < 7:
            continue
        b = list(map(int, block))
        p1, p2, d1, p3, d2, d3, d4 = b
        s1 = (p1 + d1 + d2 + d4) % 2
        s2 = (p2 + d1 + d3 + d4) % 2
        s3 = (p3 + d2 + d3 + d4) % 2
        error_pos = s1 * 1 + s2 * 2 + s3 * 4
        if error_pos != 0:
            corrected += 1
            b[error_pos - 1] ^= 1
        decoded += f"{b[2]}{b[4]}{b[5]}{b[6]}"
    return decoded, corrected


# =======Add Noise & Fading ======


def add_noise_to_signal(time_values, signal, SNR_dB, ref_power=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    # Use a fixed reference power for noise (pre-fade)
    if ref_power is None:
        ref_power = np.mean(signal**2)
    noise_power = ref_power / (10 ** (SNR_dB / 10))
    noise = np.sqrt(noise_power) * rng.standard_normal(len(signal))
    return time_values, signal + noise


def apply_fading(signal, model="none", K_factor=0, doppler_rate=0.01):
    """
    Apply realistic flat fading with unit average power and time correlation.

    model: "none", "rayleigh", or "rician"
    K_factor: Rician K-factor (linear). Use 0 for Rayleigh.
    doppler_rate: ~fraction of samples over which the channel noticeably changes.
                  Smaller -> slower fading (more correlation).
    """
    N = len(signal)
    if model is None or model.lower() == "none" or N == 0:
        return signal

    import numpy as np

    # ---- Build a correlated complex Gaussian process g[n] ----
    # Target: E[|g|^2] = 1 (unit power), AR(1) with coefficient rho.
    # Choose rho from doppler_rate so that correlation length ~ 1/doppler_rate.
    dr = float(max(1e-4, min(1.0, doppler_rate)))
    corr_len = 1.0 / dr
    rho = np.exp(-1.0 / corr_len)  # e.g., dr=0.01 -> rho≈0.99

    # Complex white noise (unit variance per complex sample)
    w = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2.0)

    g = np.empty(N, dtype=np.complex128)
    # Start in stationarity: Var(g) = 1 => scale initial sample accordingly
    g[0] = w[0] / np.sqrt(1 - rho**2)
    for n in range(1, N):
        g[n] = rho * g[n - 1] + np.sqrt(1 - rho**2) * w[n]

    # Normalize to unit power (guard against numerical drift)
    g /= np.sqrt(np.mean(np.abs(g) ** 2) + 1e-12)

    mdl = model.lower()

    if mdl == "rayleigh":
        # Zero-mean complex Gaussian -> Rayleigh envelope
        h = g  # E[|h|^2] = 1
        fading_envelope = np.abs(h)  # Rayleigh with Ω=1
        return signal * fading_envelope

    elif mdl == "rician":
        # LOS + scattered: set E[|h|^2] = 1 with given K
        K = float(max(0.0, K_factor))
        mu = np.sqrt(K / (K + 1.0))  # LOS amplitude (real, w.l.o.g.)
        sc = 1.0 / np.sqrt(K + 1.0)  # scatter scaling so power sums to 1
        h = mu + sc * g  # complex Rician process
        # (No further normalization; already unit average power.)
        fading_envelope = np.abs(h)
        return signal * fading_envelope

    else:
        raise ValueError("Unknown fading model. Use 'none', 'rayleigh', or 'rician'.")


# =======Convolution and Shift Keying Functions


def bandpass_filter(
    signal_time_domain,
    signal,
    sampling_frequency,
    low_cutoff,
    high_cutoff,
    show_plots=False,
):
    """
    Apply a square bandpass filter to a signal in the frequency domain.
    """
    N = len(signal)

    # FFT, apply filter mask, then inverse FFT
    freqs = np.fft.fftfreq(N, 1 / sampling_frequency)
    spectrum = np.fft.fft(signal)
    low_cutoff_array = np.abs(freqs) >= low_cutoff
    high_cutoff_array = np.abs(freqs) <= high_cutoff
    filter_mask = np.logical_and(low_cutoff_array, high_cutoff_array).astype(float)
    filtered_spectrum = spectrum * filter_mask
    filtered_signal = np.fft.ifft(filtered_spectrum).real

    if show_plots:
        plt.figure(figsize=(14, 12))

        # Plot original and filtered signals
        plt.subplot(5, 1, 1)
        plt.plot(signal_time_domain, signal, label="Clean Signal")
        plt.title("Original Signal")
        plt.grid(True)
        plt.legend()

        plt.subplot(5, 1, 4)
        plt.plot(freqs[: N // 2], filter_mask[: N // 2])
        plt.title("Bandpass Filter Mask")
        plt.grid(True)

        plt.subplot(5, 1, 5)
        plt.plot(
            signal_time_domain, filtered_signal, label="Filtered Signal", color="green"
        )
        plt.title("Filtered Signal")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    return filtered_signal, freqs, filter_mask


def frequency_shift_keying(
    bit_string,
    carrier_amplitude=1.0,
    frequency_deviation=100,
    carrier_frequency=1000,
    bit_rate=100,
    sampling_frequency=10000,
    create_plot=True,
):
    """
    Perform binary Frequency Shift Keying (FSK) modulation of a binary message.
    """

    # Define frequencies for '0' and '1'
    f0 = carrier_frequency - frequency_deviation
    f1 = carrier_frequency + frequency_deviation

    # Set Baud rate and arrays
    bit_duration = 1 / bit_rate
    samples_per_bit = int(sampling_frequency * bit_duration)
    signal = np.array([])
    time_values = np.array([])
    binary_signal = np.array([])

    for i, bit in enumerate(bit_string):
        # Baud timing
        t = (
            np.arange(i * samples_per_bit, (i + 1) * samples_per_bit)
            / sampling_frequency
        )

        # Wave creation
        freq = f1 if bit == "1" else f0
        bit_wave = carrier_amplitude * np.cos(2 * np.pi * freq * t)
        signal = np.concatenate((signal, bit_wave))
        time_values = np.concatenate((time_values, t))

        # Generate binary waveform (for visual reference)
        binary_wave = np.ones(samples_per_bit) * int(bit)
        binary_signal = np.concatenate((binary_signal, binary_wave))

    # Plot
    if create_plot:
        plt.figure(figsize=(12, 6))
        plt.plot(time_values, signal, label="FSK Signal")
        plt.plot(time_values, binary_signal, "r", alpha=0.5, label="Binary Input")
        plt.title(f"FSK Modulated Signal ({len(bit_string)} bits)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.ylim(carrier_amplitude * 0.1)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return time_values, signal


# =======DEMODULATION FUNCTIONS=======


def FSK_demodulation(
    signal_time_domain,
    signal,
    baud_rate,
    carrier_frequency,
    FSK_variation,
    sampling_frequency,
    original_message,
):
    """
    Fast vectorized FSK demodulation using correlation detection.
    """

    # Define frequencies for '0' and '1'
    f0 = carrier_frequency - FSK_variation
    f1 = carrier_frequency + FSK_variation

    # Compute timing and array sizes
    bit_duration = 1 / baud_rate
    samples_per_bit = int(sampling_frequency * bit_duration)
    num_bits = len(signal) // samples_per_bit
    decoded_bits = []

    # Trim signal to an integer number of bits
    signal = signal[: num_bits * samples_per_bit]
    time_values = np.arange(samples_per_bit) / sampling_frequency
    """
    # Reshape into [num_bits, samples_per_bit]
    bits_matrix = signal.reshape(num_bits, samples_per_bit)

    # Generate reference carriers for correlation
    ref0 = np.cos(2 * np.pi * f0 * time_values)
    ref1 = np.cos(2 * np.pi * f1 * time_values)

    # Compute correlation for all bits at once
    corr0 = np.dot(bits_matrix, ref0)
    corr1 = np.dot(bits_matrix, ref1)

    # Decide based on which correlation is stronger
    decoded_bits = (corr0 > corr1).astype(int)
    """

    for i in range(num_bits):
        start = i * samples_per_bit
        end = (i + 1) * samples_per_bit

        bit_slice = signal[start:end]
        time_slice = time_values[start:end]

        # Bandpass filter around f0 and f1
        filtered_low, _, _ = bandpass_filter(
            time_slice,
            bit_slice,
            sampling_frequency,
            f0 - FSK_variation,
            f0 + FSK_variation,
        )
        filtered_high, _, _ = bandpass_filter(
            time_slice,
            bit_slice,
            sampling_frequency,
            f1 - FSK_variation,
            f1 + FSK_variation,
        )

        # Compute power (energy per bit)
        low_power = np.mean(filtered_low**2)
        high_power = np.mean(filtered_high**2)

        decoded_bits.append(1 if high_power > low_power else 0)

    bit_string = "".join(str(b) for b in decoded_bits)

    # Compute Bit Error Rate
    orig_bits = np.array(original_message[: len(decoded_bits)])
    bit_error_rate = np.mean(orig_bits != decoded_bits)

    return bit_string, bit_error_rate


# ==========================================================
# =============== MAIN TRANSMISSION PROGRAM ================
# ==========================================================


def interactive_console():
    print("\n" + "=" * 60)
    print("🎛️  INTERACTIVE COMMUNICATION CHANNEL CONSOLE")
    print("=" * 60)

    # --- Mode selection ---
    mode_type = inquirer.select(
        message="Select operation mode:",
        choices=["Single Test Run", "Parameter Sweep (Multiple Simulations)"],
        default="Single Test Run",
    ).execute()

    if "Single" in mode_type:
        single_test_run()
    else:
        sweep_mode_run()


# ==========================================================
# =============== SINGLE TEST IMPLEMENTATION ===============
# ==========================================================

def single_test_run():
    """Run a single FSK simulation interactively with user-defined retransmission policy."""
    p = ask_simulation_params(mode="single")

    # Auto adjustment
    if p["auto_adjust"]:
        sampling_rate, FSK_deviation, carrier_adj = check_and_adjust_sampling(
            p["baud_rate"],
            p["sampling_rate"],
            p["carrier_frequency"] / 15,
            p["carrier_frequency"],
            sweep_mode=False,
        )
    else:
        sampling_rate = p["sampling_rate"]
        FSK_deviation = int(p["carrier_frequency"] / 15)
        carrier_adj = p["carrier_frequency"]

    print("\n🚀 Running single simulation... Please wait...\n")

    run_simulation(
        mode_choice=p["mode_choice"],
        baud_rate=p["baud_rate"],
        sampling_rate=sampling_rate,
        carrier_frequency=carrier_adj,
        SNR=p["SNR"],
        message=p["message"],
        FSK_deviation=FSK_deviation,
        error_method=p["error_method"],
        retransmit_enabled=p["retransmit_enabled"],
        fading_model=p["fading_model"],
        show_summary=True,
        sweep_mode=False,
    )

# ==========================================================
# ================ SWEEP MODE IMPLEMENTATION ===============
# ==========================================================

def sweep_mode_run():
    """Run multiple FSK simulations with parameter sweeps, including retransmission control."""
    p = ask_simulation_params(mode="sweep")

    rprint(
        f"\n[cyan]🔄 Your message is: {len(hex_to_binary_raw(string_to_hex_array(p['message'])))} bits long[/cyan]\n"
    )
    rprint("[bold cyan]🔄 Starting parameter sweep...[/bold cyan]\n")

    results = []
    total_runs = (
        len(p["baud_rates"])
        * len(p["sampling_rates"])
        * len(p["carrier_frequencies"])
        * len(p["SNR_values"])
    )
    run_count = 1

    for baud in p["baud_rates"]:
        for sample in p["sampling_rates"]:
            for carrier in p["carrier_frequencies"]:
                for snr in p["SNR_values"]:
                    rprint(
                        f"[white on blue]▶️ Run {run_count}/{total_runs}[/white on blue] → "
                        f"[cyan]Baud={baud}[/cyan], [yellow]Fs={sample}[/yellow], "
                        f"[magenta]Fc={carrier}[/magenta], [green]SNR={snr} dB[/green]"
                    )

                    start_time = time.time()

                    # Auto adjust per combination
                    if p["auto_adjust"]:
                        old_fc, old_fs = carrier, sample
                        sample, FSK_dev, carrier_adj = check_and_adjust_sampling(
                            baud, sample, carrier / 15, carrier, sweep_mode=True
                        )
                        if carrier_adj != old_fc or sample != old_fs:
                            rprint(
                                f"[bold yellow]⚙️ Adjusted:[/bold yellow] Fs → {sample}, Fc → {carrier_adj}, Δf → {FSK_dev}"
                            )
                            sleep(0.3)
                        else:
                            FSK_dev = int(carrier / 15)
                            carrier_adj = carrier
                    else:
                        FSK_dev = int(carrier / 15)
                        carrier_adj = carrier

                    raw_BER, retrans, failed_bytes = run_simulation(
                        p["mode_choice"],
                        baud,
                        sample,
                        carrier_adj,
                        snr,
                        p["message"],
                        FSK_dev,
                        error_method=p["error_method"],
                        retransmit_enabled=p["retransmit_enabled"],
                        fading_model=p["fading_model"],
                        show_summary=False,
                        sweep_mode=True,
                    )

                    elapsed = time.time() - start_time
                    results.append(
                        [
                            baud,
                            sample,
                            carrier_adj,
                            FSK_dev,
                            snr,
                            f"{raw_BER:.6f}",
                            retrans,
                            failed_bytes,
                            f"{elapsed:.2f}s",
                        ]
                    )
                    run_count += 1
                    sleep(0.25)

    display_results_table(results)

# ==========================================================
# ================== RUN SIMULATION CORE ===================
# ==========================================================

def run_simulation(
    mode_choice,
    baud_rate,
    sampling_rate,
    carrier_frequency,
    SNR,
    message,
    FSK_deviation,
    error_method="None",
    retransmit_enabled=True,
    fading_model="None",
    show_summary=True,
    sweep_mode=False,
):
    """
    UART-style byte-by-byte transmission simulation.
    - Tracks RAW BER (before retransmission or correction).
    - Retransmission affects only link reliability, not BER count.
    """

    tx_hex = string_to_hex_array(message)
    tx_bits = hex_to_binary_raw(tx_hex)
    received_bits = ""
    total_retransmissions = 0
    total_bytes_with_errors = 0

    # For true raw BER stats
    raw_bit_errors = 0
    raw_bits_total = 0

    # Split into bytes
    byte_chunks = [tx_bits[i : i + 8].ljust(8, "0") for i in range(0, len(tx_bits), 8)]

    for byte_bits in byte_chunks:
        # Encode
        if "Parity" in error_method:
            encoded_bits = add_parity_bit(byte_bits)
        elif "Hamming" in error_method:
            encoded_bits = hamming74_encode(byte_bits)
        else:
            encoded_bits = byte_bits

        success = False
        attempt = 0
        max_retries = 3 if retransmit_enabled else 1

        while not success and attempt < max_retries:
            attempt += 1

            t, tx_signal = frequency_shift_keying(
                bit_string=encoded_bits,
                carrier_amplitude=1.0,
                frequency_deviation=FSK_deviation,
                carrier_frequency=carrier_frequency,
                bit_rate=baud_rate,
                sampling_frequency=sampling_rate,
                create_plot=False,
            )

            # --- Channel and noise ---
            filtered_signal, _, _ = bandpass_filter(
                signal_time_domain=t,
                signal=tx_signal,
                sampling_frequency=sampling_rate,
                low_cutoff=carrier_frequency - 0.1 * carrier_frequency,
                high_cutoff=carrier_frequency + 0.1 * carrier_frequency,
            )

            # Pre-fade reference (AFTER filtering, BEFORE fading)
            ref_power = np.mean(filtered_signal**2)

            # Apply fading
            faded_signal = apply_fading(filtered_signal, model=fading_model.lower())

            # Add AWGN with variance fixed by the *pre-fade* ref power
            _, noisy_signal = add_noise_to_signal(
                t, faded_signal, SNR, ref_power=ref_power
            )

            # --- Demodulation ---
            if "FSK" in mode_choice:
                rx_bits, _ = FSK_demodulation(
                    signal_time_domain=t,
                    signal=noisy_signal,
                    baud_rate=baud_rate,
                    carrier_frequency=carrier_frequency,
                    FSK_variation=FSK_deviation,
                    sampling_frequency=sampling_rate,
                    original_message=[int(b) for b in encoded_bits],
                )
            else:
                rx_bits = encoded_bits

            # --- Raw BER collection (before correction/retransmit) ---
            if rx_bits and len(rx_bits) == len(encoded_bits):
                bit_errors = sum(
                    encoded_bits[j] != rx_bits[j] for j in range(len(encoded_bits))
                )
                raw_bit_errors += bit_errors
                raw_bits_total += len(encoded_bits)

            # --- Decode / error check ---
            if "Parity" in error_method:
                decoded_bits, parity_errors = check_parity_bit(rx_bits)
                # require BOTH: no parity error AND payload matches original
                success = (parity_errors == 0) and (decoded_bits == byte_bits)
            elif "Hamming" in error_method:
                decoded_bits, corrected_bits = hamming74_decode(rx_bits)
                success = corrected_bits <= 1
            else:
                decoded_bits = rx_bits
                if rx_bits and len(rx_bits) == len(encoded_bits):
                    ber_measured = np.mean(
                        [
                            encoded_bits[j] != rx_bits[j]
                            for j in range(len(encoded_bits))
                        ]
                    )
                    success = ber_measured == 0.0
                else:
                    success = False

            if not success and retransmit_enabled:
                total_retransmissions += 1

        if success:
            received_bits += decoded_bits
        else:
            total_bytes_with_errors += 1

    # === Compute RAW BER only ===
    raw_BER = raw_bit_errors / raw_bits_total if raw_bits_total > 0 else 0.0

    # === Integrity check ===
    message_match = tx_bits[: len(received_bits)] == received_bits

    # === Display results ===
    if show_summary and not sweep_mode:
        print("\n" + "=" * 60)
        print("📡 TRANSMISSION SUMMARY")
        print("=" * 60)
        print(f"🧩 Modulation: {mode_choice}")
        print(f"💾 Error Control: {error_method}")
        print(f"📶 Carrier Frequency: {carrier_frequency} Hz")
        print(f"⏱️ Baud Rate: {baud_rate}")
        print(f"🎚️ SNR: {SNR} dB")
        print(f"🔁 Retransmission Enabled: {retransmit_enabled}")
        print(f"💥 Raw BER: {raw_BER:.6f}")
        print(f"🔁 Retransmissions: {total_retransmissions}")
        print(f"⚠️ Bytes Failed After Retries: {total_bytes_with_errors}")
        print(f"✅ Message OK: {'Yes' if message_match else 'No'}")
        print("=" * 60 + "\n")

    return raw_BER, total_retransmissions, total_bytes_with_errors


# This just runs the console program on file run
if __name__ == "__main__":
    interactive_console()

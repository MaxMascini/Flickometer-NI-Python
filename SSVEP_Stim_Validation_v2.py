# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# HARMONIC DETECTION INTEGRATION
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import nidaqmx
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import linregress
from nidaqmx.constants import TerminalConfiguration
import time
import csv

# Configuration
SAMPLE_RATE = 2000  # Sampling rate in Hz
SAMPLES_PER_CHANNEL = 8000  # Number of samples to acquire
CHANNELS = ["Dev1/ai1", "Dev1/ai2", "Dev1/ai3", "Dev1/ai4", "Dev1/ai5", "Dev1/ai6"]
FREQ_RANGE = (9, 14)  # Frequency range for band-pass filter
AMPLITUDE_THRESHOLD = 0.01  # Minimum amplitude threshold for valid signal
HARMONIC_COUNT = 3  # Number of harmonics to analyze

# Band-pass filter function
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# Frequency and harmonic analysis function
def analyze_frequency_and_harmonics(signal, sample_rate):
    fft_length = len(signal) * 10  # Zero-padding to increase resolution
    fft_result = np.fft.fft(signal, n=fft_length)
    frequencies = np.fft.fftfreq(fft_length, d=1/sample_rate)
    positive_frequencies = frequencies[:fft_length//2]
    fft_magnitude = np.abs(fft_result[:fft_length//2])

    # Find dominant frequency
    dominant_index = np.argmax(fft_magnitude)
    fundamental_freq = positive_frequencies[dominant_index]
    fundamental_magnitude = fft_magnitude[dominant_index]

    # Analyze harmonics
    harmonics = {}
    for n in range(2, HARMONIC_COUNT + 1):
        harmonic_freq = fundamental_freq * n
        if harmonic_freq > max(positive_frequencies):
            break
        harmonic_index = np.argmin(np.abs(positive_frequencies - harmonic_freq))
        harmonics[f"{n}x"] = {
            "frequency": positive_frequencies[harmonic_index],
            "fft_magnitude": fft_magnitude[harmonic_index],
        }

    return fundamental_freq, fundamental_magnitude, harmonics

# SNR calculation
def calculate_snr(signal):
    signal_power = np.mean(np.square(signal))
    noise_power = np.var(signal - np.mean(signal))
    return 10 * np.log10(signal_power / noise_power)

# Main measurement function
def measure_frequencies():
    with nidaqmx.Task() as task:
        # Add analog input channels with proper configuration
        for channel in CHANNELS:
            task.ai_channels.add_ai_voltage_chan(
                channel,
                terminal_config=TerminalConfiguration.RSE,
                min_val=0.0,  # Adjust based on expected range
                max_val=4.5   # Adjust based on expected range
            )

        # Configure sample clock
        task.timing.cfg_samp_clk_timing(SAMPLE_RATE, samps_per_chan=SAMPLES_PER_CHANNEL)

        print("Starting SSVEP validation. Press Ctrl+C to stop.")

        # Open a CSV file to log data
        with open("ssvep_validation_log.csv", "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            # Write header row
            header = ["Channel", "Fundamental Frequency (Hz)", "Fundamental FFT Magnitude"]
            for n in range(2, HARMONIC_COUNT + 1):
                header.append(f"Harmonic {n}x Frequency (Hz)")
                header.append(f"Harmonic {n}x FFT Magnitude")
            header.extend(["Stability (Variance)", "Amplitude", "SNR (dB)", "Frequency Drift Slope (Hz/reading)"])
            csvwriter.writerow(header)

            try:
                frequency_history = {channel: [] for channel in CHANNELS}
                amplitude_history = {channel: [] for channel in CHANNELS}

                while True:
                    data = np.array(task.read(number_of_samples_per_channel=SAMPLES_PER_CHANNEL))

                    # Process data and calculate frequencies
                    for i, signal in enumerate(data):
                        # Apply band-pass filter
                        filtered_signal = bandpass_filter(signal, FREQ_RANGE[0], FREQ_RANGE[1], SAMPLE_RATE)
                        amplitude = np.max(filtered_signal) - np.min(filtered_signal)

                        if amplitude < AMPLITUDE_THRESHOLD:
                            print(f"Channel {CHANNELS[i]}: Weak Signal Detected (Amplitude = {amplitude:.3f})")
                            continue

                        # Analyze frequency and harmonics
                        fundamental_freq, fundamental_magnitude, harmonics = analyze_frequency_and_harmonics(filtered_signal, SAMPLE_RATE)

                        # Record frequency and amplitude for stability analysis
                        frequency_history[CHANNELS[i]].append(fundamental_freq)
                        amplitude_history[CHANNELS[i]].append(amplitude)
                        if len(frequency_history[CHANNELS[i]]) > 20:  # Keep the last 20 readings
                            frequency_history[CHANNELS[i]] = frequency_history[CHANNELS[i]][-20:]
                            amplitude_history[CHANNELS[i]] = amplitude_history[CHANNELS[i]][-20:]

                        # Calculate stability (variance) and CV
                        variance = np.var(frequency_history[CHANNELS[i]])
                        mean_freq = np.mean(frequency_history[CHANNELS[i]])
                        cv = np.sqrt(variance) / mean_freq  # Coefficient of Variation

                        # Calculate frequency drift
                        time_points = np.arange(len(frequency_history[CHANNELS[i]]))
                        slope, _, _, _, _ = linregress(time_points, frequency_history[CHANNELS[i]])

                        # Calculate SNR
                        snr = calculate_snr(filtered_signal)

                        # Output results
                        print(f"Channel {CHANNELS[i]:<10}:")
                        print(f"  Fundamental Frequency: {fundamental_freq:.2f} Hz (FFT Magnitude: {fundamental_magnitude:.3f})")
                        row = [CHANNELS[i], fundamental_freq, fundamental_magnitude]

                        for harmonic, info in harmonics.items():
                            print(f"  Harmonic {harmonic}: {info['frequency']:.2f} Hz (FFT Magnitude: {info['fft_magnitude']:.3f})")
                            row.extend([info['frequency'], info['fft_magnitude']])

                        print(f"  Stability (Variance): {variance:.3f}")
                        print(f"  Coefficient of Variation (CV): {cv:.3f}")
                        print(f"  Frequency Drift Slope: {slope:.6f} Hz/reading")
                        print(f"  Amplitude: {amplitude:.3f}")
                        print(f"  SNR: {snr:.3f} dB")
                        row.extend([variance, amplitude, snr, slope])

                        # Write data to CSV
                        csvwriter.writerow(row)

                    print("-" * 50)
                    time.sleep(0.5)

            except KeyboardInterrupt:
                print("SSVEP validation stopped.")

if __name__ == "__main__":
    measure_frequencies()

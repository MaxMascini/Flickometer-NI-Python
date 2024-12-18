# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# HARMONIC DETECTION INTEGRATION
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import nidaqmx
import numpy as np
from scipy.signal import butter, filtfilt
from nidaqmx.constants import TerminalConfiguration
import time

# Configuration
SAMPLE_RATE = 1500  # Sampling rate in Hz
SAMPLES_PER_CHANNEL = 5000  # Number of samples to acquire
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
    fundamental_strength = fft_magnitude[dominant_index]

    # Analyze harmonics
    harmonics = {}
    for n in range(2, HARMONIC_COUNT + 1):
        harmonic_freq = fundamental_freq * n
        if harmonic_freq > max(positive_frequencies):
            break
        harmonic_index = np.argmin(np.abs(positive_frequencies - harmonic_freq))
        harmonics[f"{n}x"] = {
            "frequency": positive_frequencies[harmonic_index],
            "strength": fft_magnitude[harmonic_index],
        }

    return fundamental_freq, fundamental_strength, harmonics

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
        try:
            frequency_history = {channel: [] for channel in CHANNELS}

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
                    fundamental_freq, fundamental_strength, harmonics = analyze_frequency_and_harmonics(filtered_signal, SAMPLE_RATE)

                    # Record frequency for stability analysis
                    frequency_history[CHANNELS[i]].append(fundamental_freq)
                    if len(frequency_history[CHANNELS[i]]) > 10:  # Keep the last 10 readings
                        frequency_history[CHANNELS[i]] = frequency_history[CHANNELS[i]][-10:]

                    # Calculate stability (variance over time)
                    stability = np.var(frequency_history[CHANNELS[i]])

                    # Output results
                    print(f"Channel {CHANNELS[i]:<10}:")
                    print(f"  Fundamental Frequency: {fundamental_freq:.2f} Hz (Strength: {fundamental_strength:.3f})")
                    for harmonic, info in harmonics.items():
                        print(f"  Harmonic {harmonic}: {info['frequency']:.2f} Hz (Strength: {info['strength']:.3f})")
                    print(f"  Stability (Variance): {stability:.3f}")
                    print(f"  Amplitude: {amplitude:.3f}")

                print("-" * 50)
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("SSVEP validation stopped.")

if __name__ == "__main__":
    measure_frequencies()

import nidaqmx
import numpy as np
from scipy.signal import butter, filtfilt
import time

# Configuration
SAMPLE_RATE = 1500  # Sampling rate in Hz
SAMPLES_PER_CHANNEL = 5000  # Number of samples to acquire
CHANNELS = ["Dev1/ai1", "Dev1/ai2", "Dev1/ai3", "Dev1/ai4", "Dev1/ai5", "Dev1/ai6"]
FREQ_RANGE = (9, 14)  # Frequency range for band-pass filter
HIGH_PASS_CUTOFF = 1.0  # High-pass filter cutoff frequency in Hz
AMPLITUDE_THRESHOLD = 0.01  # Minimum amplitude threshold for valid signal

# Band-pass filter function
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# High-pass filter function
def highpass_filter(signal, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    high = cutoff / nyquist
    b, a = butter(order, high, btype='high')
    return filtfilt(b, a, signal)

# Drift correction function
def remove_drift(signal):
    """Remove linear drift from the signal."""
    x = np.arange(len(signal))
    p = np.polyfit(x, signal, 1)  # Fit a linear trend
    trend = np.polyval(p, x)  # Generate the trend line
    return signal - trend

# Frequency calculation function
def calculate_dominant_frequency(signal, sample_rate):
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), d=1/sample_rate)
    positive_frequencies = frequencies[:len(frequencies)//2]
    fft_magnitude = np.abs(fft_result[:len(frequencies)//2])
    dominant_index = np.argmax(fft_magnitude)
    return positive_frequencies[dominant_index]

# Main measurement function
def measure_frequencies():
    with nidaqmx.Task() as task:
        # Add analog input channels
        
        from nidaqmx.constants import TerminalConfiguration

# Add analog input channels with specific configuration
        for channel in CHANNELS:
            task.ai_channels.add_ai_voltage_chan(
                physical_channel=channel,
                terminal_config=TerminalConfiguration.RSE,  # Change to match your wiring setup
                min_val=0.0,  # Adjust to your signal range
                max_val=5.0,  # Adjust to your signal range
            )

        
        # for channel in CHANNELS:
        #     task.ai_channels.add_ai_voltage_chan(channel, min_val=-10.0, max_val=10.0)

        # Configure sample clock
        task.timing.cfg_samp_clk_timing(SAMPLE_RATE, samps_per_chan=SAMPLES_PER_CHANNEL)

        print("Starting continuous data acquisition. Press Ctrl+C to stop.")
        try:
            while True:
                data = np.array(task.read(number_of_samples_per_channel=SAMPLES_PER_CHANNEL))

                # Process data and calculate frequencies
                frequencies = []
                for i, signal in enumerate(data):
                    # Log raw signal amplitude for debugging
                    raw_amplitude = np.max(signal) - np.min(signal)
                    print(f"Channel {CHANNELS[i]}: Raw Signal Min = {np.min(signal):.3f}, Max = {np.max(signal):.3f}, Amplitude = {raw_amplitude:.3f}")

                    # Bypass filtering temporarily for debugging
                    freq_raw = calculate_dominant_frequency(signal, SAMPLE_RATE)
                    print(f"Channel {CHANNELS[i]}: Dominant Frequency (Raw Signal) = {freq_raw:.3f} Hz")

                    # High-pass filter to remove very low-frequency components
                    highpassed_signal = highpass_filter(signal, HIGH_PASS_CUTOFF, SAMPLE_RATE)
                    # Remove drift
                    drift_corrected_signal = remove_drift(highpassed_signal)
                    # Apply band-pass filter
                    filtered_signal = bandpass_filter(drift_corrected_signal, FREQ_RANGE[0], FREQ_RANGE[1], SAMPLE_RATE)
                    # Check amplitude after filtering
                    amplitude = np.max(filtered_signal) - np.min(filtered_signal)
                    if amplitude < AMPLITUDE_THRESHOLD:
                        freq = 0.0  # Signal too weak to determine frequency
                    else:
                        # Calculate dominant frequency
                        freq = calculate_dominant_frequency(filtered_signal, SAMPLE_RATE)
                    frequencies.append(freq)
                    print(f"Channel {CHANNELS[i]:<10}: Dominant Frequency (Filtered) = {freq:.3f} Hz (Amplitude: {amplitude:.3f})")

                print("-" * 50)  # Separator for better readability
                time.sleep(0.5)  # Delay between updates

        except KeyboardInterrupt:
            print("Data acquisition stopped.")

if __name__ == "__main__":
    measure_frequencies()

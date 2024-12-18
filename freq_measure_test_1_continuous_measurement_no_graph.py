import nidaqmx
import numpy as np
from scipy.signal import butter, filtfilt
import time

# Configuration
SAMPLE_RATE = 1500  # Sampling rate in Hz
SAMPLES_PER_CHANNEL = 5000  # Number of samples to acquire
CHANNELS = ["Dev1/ai1", "Dev1/ai2", "Dev1/ai3", "Dev1/ai4", "Dev1/ai5", "Dev1/ai6"]
FREQ_RANGE = (9, 14)  # Frequency range for band-pass filter

# Band-pass filter function
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

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
        for channel in CHANNELS:
            task.ai_channels.add_ai_voltage_chan(channel, min_val=-10.0, max_val=10.0)

        # Configure sample clock
        task.timing.cfg_samp_clk_timing(SAMPLE_RATE, samps_per_chan=SAMPLES_PER_CHANNEL)

        print("Starting continuous data acquisition. Press Ctrl+C to stop.")
        try:
            while True:
                data = np.array(task.read(number_of_samples_per_channel=SAMPLES_PER_CHANNEL))

                # Process data and calculate frequencies
                frequencies = []
                for i, signal in enumerate(data):
                    # Apply band-pass filter
                    filtered_signal = bandpass_filter(signal, FREQ_RANGE[0], FREQ_RANGE[1], SAMPLE_RATE)
                    # Calculate dominant frequency
                    freq = calculate_dominant_frequency(filtered_signal, SAMPLE_RATE)
                    frequencies.append(freq)
                    print(f"Channel {CHANNELS[i]}: Dominant Frequency = {freq:.2f} Hz")

                time.sleep(0.5)  # Delay between updates

        except KeyboardInterrupt:
            print("Data acquisition stopped.")

if __name__ == "__main__":
    measure_frequencies()
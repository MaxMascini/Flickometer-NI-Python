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

# Band-pass filter function
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# Frequency calculation function with finer resolution
def calculate_dominant_frequency(signal, sample_rate):
    fft_length = len(signal) * 10  # Zero-padding to increase resolution
    fft_result = np.fft.fft(signal, n=fft_length)
    frequencies = np.fft.fftfreq(fft_length, d=1/sample_rate)
    positive_frequencies = frequencies[:fft_length//2]
    fft_magnitude = np.abs(fft_result[:fft_length//2])
    dominant_index = np.argmax(fft_magnitude)
    return positive_frequencies[dominant_index]

# Main measurement function
def measure_frequencies():
    with nidaqmx.Task() as task:
        # Add analog input channels with proper configuration
        for channel in CHANNELS:
            task.ai_channels.add_ai_voltage_chan(
                channel,
                terminal_config=TerminalConfiguration.RSE,  # Adjust to your setup
                min_val=0.0,  # Adjust based on expected range
                max_val=2.0   # Adjust based on expected range
            )

        # Configure sample clock
        task.timing.cfg_samp_clk_timing(SAMPLE_RATE, samps_per_chan=SAMPLES_PER_CHANNEL)

        print("Starting continuous data acquisition. Press Ctrl+C to stop.")
        try:
            while True:
                data = np.array(task.read(number_of_samples_per_channel=SAMPLES_PER_CHANNEL))

                # Process data and calculate frequencies
                for i, signal in enumerate(data):
                    # Apply band-pass filter
                    filtered_signal = bandpass_filter(signal, FREQ_RANGE[0], FREQ_RANGE[1], SAMPLE_RATE)
                    amplitude = np.max(filtered_signal) - np.min(filtered_signal)

                    if amplitude < AMPLITUDE_THRESHOLD:
                        print(f"Channel {CHANNELS[i]}: Weak Signal Detected (Amplitude = {amplitude:.3f})")
                        freq = 0.0
                    else:
                        freq = calculate_dominant_frequency(filtered_signal, SAMPLE_RATE)
                        print(f"Channel {CHANNELS[i]:<10}: Dominant Frequency = {freq:.2f} Hz (Amplitude: {amplitude:.3f})")

                print("-" * 50)
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("Data acquisition stopped.")

if __name__ == "__main__":
    measure_frequencies()


# import nidaqmx
# import numpy as np
# from scipy.signal import butter, filtfilt
# from nidaqmx.constants import TerminalConfiguration
# import time

# # Configuration
# SAMPLE_RATE = 1500  # Sampling rate in Hz
# SAMPLES_PER_CHANNEL = 5000  # Number of samples to acquire
# CHANNELS = ["Dev1/ai1", "Dev1/ai2", "Dev1/ai3", "Dev1/ai4", "Dev1/ai5", "Dev1/ai6"]
# FREQ_RANGE = (9, 14)  # Frequency range for band-pass filter
# AMPLITUDE_THRESHOLD = 0.01  # Minimum amplitude threshold for valid signal

# # Band-pass filter function
# def bandpass_filter(signal, lowcut, highcut, fs, order=4):
#     nyquist = 0.5 * fs
#     low = lowcut / nyquist
#     high = highcut / nyquist
#     b, a = butter(order, [low, high], btype='band')
#     return filtfilt(b, a, signal)

# # Frequency calculation function with finer resolution
# def calculate_dominant_frequency(signal, sample_rate):
#     fft_length = len(signal) * 10  # Zero-padding to increase resolution
#     fft_result = np.fft.fft(signal, n=fft_length)
#     frequencies = np.fft.fftfreq(fft_length, d=1/sample_rate)
#     positive_frequencies = frequencies[:fft_length//2]
#     fft_magnitude = np.abs(fft_result[:fft_length//2])
#     dominant_index = np.argmax(fft_magnitude)
#     return positive_frequencies[dominant_index]

# # Main measurement function
# def measure_frequencies():
#     with nidaqmx.Task() as task:
#         # Add analog input channels with proper configuration
#         for channel in CHANNELS:
#             task.ai_channels.add_ai_voltage_chan(
#                 channel,
#                 terminal_config=TerminalConfiguration.RSE,
#                 min_val=0.0,  # Adjust based on expected range
#                 max_val=2.0   # Adjust based on expected range
#             )

#         # Configure sample clock
#         task.timing.cfg_samp_clk_timing(SAMPLE_RATE, samps_per_chan=SAMPLES_PER_CHANNEL)

#         print("Starting SSVEP validation. Press Ctrl+C to stop.")
#         try:
#             while True:
#                 data = np.array(task.read(number_of_samples_per_channel=SAMPLES_PER_CHANNEL))

#                 # Process data and calculate frequencies
#                 for i, signal in enumerate(data):
#                     # Apply band-pass filter
#                     filtered_signal = bandpass_filter(signal, FREQ_RANGE[0], FREQ_RANGE[1], SAMPLE_RATE)
#                     amplitude = np.max(filtered_signal) - np.min(filtered_signal)

#                     if amplitude < AMPLITUDE_THRESHOLD:
#                         print(f"Channel {CHANNELS[i]}: Weak Signal Detected (Amplitude = {amplitude:.3f})")
#                         freq = 0.0
#                     else:
#                         freq = calculate_dominant_frequency(filtered_signal, SAMPLE_RATE)
#                         print(f"Channel {CHANNELS[i]:<10}: Dominant Frequency = {freq:.2f} Hz (Amplitude: {amplitude:.3f})")

#                 print("-" * 50)
#                 time.sleep(0.5)

#         except KeyboardInterrupt:
#             print("SSVEP validation stopped.")

# if __name__ == "__main__":
#     measure_frequencies()



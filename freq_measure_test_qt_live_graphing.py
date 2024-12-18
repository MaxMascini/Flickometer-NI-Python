import nidaqmx
import numpy as np
from scipy.signal import butter, filtfilt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QWidget, QHBoxLayout
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
import sys

# Configuration
SAMPLE_RATE = 1000  # Sampling rate in Hz
SAMPLES_PER_CHANNEL = 2000  # Number of samples per update
CHANNELS = ["Dev1/ai1", "Dev1/ai2", "Dev1/ai3", "Dev1/ai4", "Dev1/ai5", "Dev1/ai6"]
FREQ_RANGE = (9, 14)  # Expected frequency range for filtering

class FrequencyAnalyzer:
    """
    Handles data acquisition and frequency calculation.
    """
    def __init__(self):
        self.task = nidaqmx.Task()
        for channel in CHANNELS:
            self.task.ai_channels.add_ai_voltage_chan(channel, min_val=-10.0, max_val=10.0)
        self.task.timing.cfg_samp_clk_timing(SAMPLE_RATE, samps_per_chan=SAMPLES_PER_CHANNEL)

    def get_data(self):
        """Acquire data from the channels."""
        return np.array(self.task.read(number_of_samples_per_channel=SAMPLES_PER_CHANNEL))

    def calculate_frequencies(self, data):
        """Calculate the dominant frequency for each channel."""
        frequencies = []
        for signal in data:
            filtered_signal = self.bandpass_filter(signal, FREQ_RANGE[0], FREQ_RANGE[1], SAMPLE_RATE)
            frequencies.append(self.get_dominant_frequency(filtered_signal))
        return frequencies

    def bandpass_filter(self, signal, lowcut, highcut, fs, order=4):
        """Apply a bandpass filter to the signal."""
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)

    def get_dominant_frequency(self, signal):
        """Find the dominant frequency using FFT."""
        fft_result = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(signal), d=1/SAMPLE_RATE)
        positive_frequencies = frequencies[:len(frequencies)//2]
        fft_magnitude = np.abs(fft_result[:len(frequencies)//2])
        dominant_index = np.argmax(fft_magnitude)
        return positive_frequencies[dominant_index]

    def close(self):
        self.task.close()

class FrequencyDisplay(QMainWindow):
    """
    GUI for displaying live frequency and graphs.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Frequency and Signal Display")
        self.setGeometry(100, 100, 800, 600)

        # Main Layout
        self.central_widget = QWidget()
        self.main_layout = QVBoxLayout()

        # Frequency Labels
        self.label_layout = QHBoxLayout()
        self.labels = [QLabel(f"Photodiode {i+1}: - Hz") for i in range(len(CHANNELS))]
        for label in self.labels:
            self.label_layout.addWidget(label)
        self.main_layout.addLayout(self.label_layout)

        # Graphs
        self.graphs = []
        self.graph_layout = QVBoxLayout()
        for i in range(len(CHANNELS)):
            plot = pg.PlotWidget()
            plot.setYRange(-10, 10)  # Adjust based on expected signal amplitude
            plot.setTitle(f"Photodiode {i+1}")
            self.graphs.append(plot)
            self.graph_layout.addWidget(plot)
        self.main_layout.addLayout(self.graph_layout)

        # Set central widget
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)

        # Frequency Analyzer
        self.analyzer = FrequencyAnalyzer()

        # Timer for live updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(500)  # Update every 500 ms

    def update_display(self):
        try:
            data = self.analyzer.get_data()
            frequencies = self.analyzer.calculate_frequencies(data)
            for i, freq in enumerate(frequencies):
                self.labels[i].setText(f"Photodiode {i+1}: {freq:.2f} Hz")
                self.graphs[i].plot(data[i], clear=True)  # Plot signal data
        except Exception as e:
            for label in self.labels:
                label.setText("Error reading frequency.")
            print(f"Error: {e}")

    def closeEvent(self, event):
        self.analyzer.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = FrequencyDisplay()
    main_window.show()
    sys.exit(app.exec_())

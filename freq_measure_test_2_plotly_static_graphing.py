import nidaqmx
import numpy as np
import plotly.graph_objects as go
from nidaqmx.constants import TerminalConfiguration


# Configuration
SAMPLE_RATE = 1000  # Sampling rate in Hz
SAMPLES = 5000  # Number of samples to acquire
CHANNELS = ["Dev1/ai1", "Dev1/ai2", "Dev1/ai3", "Dev1/ai4", "Dev1/ai5", "Dev1/ai6"]  # Analog input channels to read

# Main function to record and plot voltage
def record_and_plot_voltage():
    with nidaqmx.Task() as task:
        # Add analog input channels
        # for channel in CHANNELS:
        #     task.ai_channels.add_ai_voltage_chan(channel, min_val=-10.0, max_val=10.0)
        for channel in CHANNELS:
            task.ai_channels.add_ai_voltage_chan(
                channel,
                terminal_config=TerminalConfiguration.RSE,  # Adjust to your setup
                min_val=0.0,  # Adjust based on expected range
                max_val=5.0   # Adjust based on expected range
            )

        # Configure sample clock
        task.timing.cfg_samp_clk_timing(SAMPLE_RATE, samps_per_chan=SAMPLES)

        print("Acquiring data...")
        # Read voltage data
        data = np.array(task.read(number_of_samples_per_channel=SAMPLES))
        print("Data acquisition complete.")

        # Debugging: Print summary statistics for raw data
        for i, channel_data in enumerate(data):
            print(f"Channel {CHANNELS[i]}: Min={np.min(channel_data):.2f}, Max={np.max(channel_data):.2f}, StdDev={np.std(channel_data):.2f}")

        # Check for constant signals
        for i, channel_data in enumerate(data):
            if np.all(channel_data == channel_data[0]):
                print(f"Warning: Constant signal detected on {CHANNELS[i]}. Check connections.")

        # Apply offset correction (if needed)
        data = data - np.mean(data, axis=1, keepdims=True)  # Zero-center each channel

        # Generate a time axis for plotting
        time_axis = np.linspace(0, SAMPLES / SAMPLE_RATE, SAMPLES)

        # Create a Plotly figure
        fig = go.Figure()

        # Add traces for each channel
        for i, channel_data in enumerate(data):
            fig.add_trace(go.Scatter(x=time_axis, y=channel_data, mode='lines', name=f"Voltage on {CHANNELS[i]}"))

        # Update layout
        fig.update_layout(
            title="Voltage Over Time for All Channels",
            xaxis_title="Time (s)",
            yaxis_title="Voltage (V)",
            legend_title="Channels",
            template="plotly_white"
        )

        # Show the plot
        fig.show()

if __name__ == "__main__":
    record_and_plot_voltage()

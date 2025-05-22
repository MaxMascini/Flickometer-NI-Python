# live_flicker_fft.py  –  fixed read_many_sample call
import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
from nidaqmx.stream_readers import AnalogSingleChannelReader
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
import time

# ---------- USER SETTINGS ----------
DEVICE        = "Dev2"
CHANNEL       = f"{DEVICE}/ai1"
CONFIG        = TerminalConfiguration.RSE #.DIFF     # AI1+ / AI9‑
FS            = 2000            # Hz
WINDOW_SEC    = 5               # seconds per capture
BAND_MAX_HZ   = 75              # freq axis limit
# -----------------------------------

N_SAMPLES = int(FS * WINDOW_SEC)

# --- DAQ setup ------------------------------------------------------------
task = nidaqmx.Task()
task.ai_channels.add_ai_voltage_chan(
    CHANNEL, terminal_config=CONFIG, min_val=-5.0, max_val=5.0)
task.timing.cfg_samp_clk_timing(
    rate=FS,
    sample_mode=AcquisitionType.CONTINUOUS,
    samps_per_chan=N_SAMPLES * 2)          # double‑buffer

reader = AnalogSingleChannelReader(task.in_stream)
buffer = np.empty(N_SAMPLES, dtype=np.float64)
task.start()

# --- Matplotlib live figure ----------------------------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(7, 4))
line,   = ax.semilogy([], [], lw=1.2)
peak_vl = ax.axvline(0, color="red", ls="--", alpha=.7)
peak_txt = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                   ha="left", va="top", fontsize=9, color="red")
ax.set_xlim(0, BAND_MAX_HZ * 1.3)
ax.set_ylim(1e-8, 1e-2)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD (V²/Hz)")
ax.set_title("Live photodiode spectrum   -  Ctrl-C to stop")
ax.grid(True, which="both", ls=":")

try:
    while True:
        # ---- acquire one window ----
        reader.read_many_sample(
            buffer,
            number_of_samples_per_channel=N_SAMPLES,   # ← fixed
            timeout=WINDOW_SEC + 1
        )

        # ---- Welch PSD ----
        f, pxx = welch(buffer, fs=FS, nperseg=2048, scaling="density")
        sub    = f <= BAND_MAX_HZ
        peak_f = f[sub][np.argmax(pxx[sub])]

        # ---- update plot ----
        line.set_data(f, pxx)
        peak_vl.set_xdata([peak_f, peak_f])
        peak_txt.set_text(f"Peak ≈ {peak_f:5.2f} Hz")
        ax.set_ylim(pxx.min()*0.5, pxx.max()*2)
        fig.canvas.draw(); fig.canvas.flush_events()

        # ---- console read‑out ----
        print(f"{time.strftime('%H:%M:%S')}  peak ≈ {peak_f:5.2f} Hz")

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    task.close()

# """
# flicker_counter_dev2.py
# Robust frequency read‑out from a photodiode on Dev2/PFI0 using edge counts.

# ∙ Works on NI USB‑6215 or any device with a 32‑bit edge counter.
# ∙ No DAQ‑time‑out errors because we don't ask DAQmx to do frequency math.
# ∙ Resolution ~ 0.1 Hz for TARGET_MAX_HZ ≤ 100 Hz.
# """

# import time
# import nidaqmx

# DEVICE        = "Dev2"
# COUNTER_CH    = f"{DEVICE}/ctr0"
# TARGET_MAX_HZ = 75             # highest stimulus frequency you care about
# RESOLUTION_HZ = 0.1            # ±0.1 Hz single‑count resolution target

# # dwell time that gives 1 count ≈ RESOLUTION_HZ at TARGET_MAX_HZ
# READ_INTERVAL = 1 / RESOLUTION_HZ           # seconds (e.g. 10 s for 0.1 Hz)
# if READ_INTERVAL * TARGET_MAX_HZ > 2**31:   # make sure the 32‑bit counter won't overflow
#     raise ValueError("Dwell interval too long – would overflow 32‑bit counter.")

# print(f"Using dwell interval {READ_INTERVAL:.1f} s "
#       f"→ ±{RESOLUTION_HZ}Hz resolution at {TARGET_MAX_HZ}Hz")

# def main():
#     with nidaqmx.Task() as task:
#         task.ci_channels.add_ci_count_edges_chan(
#             COUNTER_CH,
#             edge=nidaqmx.constants.Edge.RISING,
#             initial_count=0,
#             count_direction=nidaqmx.constants.CountDirection.COUNT_UP,
#         )
#         task.start()

#         last_time  = time.perf_counter()
#         last_count = int(task.read())

#         print("\nPress Ctrl‑C to stop.\n")
#         while True:
#             time.sleep(READ_INTERVAL)
#             now   = time.perf_counter()
#             count = int(task.read())

#             delta_edges = count - last_count          # integer
#             delta_t     = now - last_time             # seconds (float)
#             freq        = delta_edges / delta_t if delta_t else 0.0

#             print(f"{freq:8.3f}Hz  "
#                   f"(edges {delta_edges:6d}  in {delta_t:7.3f}s)")

#             last_time  = now
#             last_count = count

# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\nStopped.")



# analog_flicker_fft.py  –  NI‑DAQmx + NumPy/SciPy/Matplotlib
# -----------------------------------------------------------
# Requirements:
#   • NI‑DAQmx runtime / driver
#   • pip install nidaqmx numpy scipy matplotlib


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

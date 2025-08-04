import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import os

# === Paths ===
base_dir = "code/ml-project/Late_Fusion_CNN"
subject_id = "01"
paths = {
    "eeg_raw": os.path.join(base_dir, f"raw_data/eeg/s{subject_id}_eeg.mat"),
    "eeg_filtered": os.path.join(base_dir, f"processed_data/eeg/continuous/s{subject_id}_eeg_cnt.npy"),
    "fnirs_raw": os.path.join(base_dir, f"raw_data/fnirs/s{subject_id}_fnirs.mat"),
    "fnirs_filtered": os.path.join(base_dir, f"processed_data/fnirs/continuous/s{subject_id}_fnirs_cnt.npy")
}

# === Load EEG ===
eeg_fs = 500
eeg_window_sec = 10
eeg_window_samples = eeg_window_sec * eeg_fs
eeg_raw = sio.loadmat(paths["eeg_raw"])["cnt_nback"][0, 0]["x"]
if eeg_raw.shape[0] < eeg_raw.shape[1]:
    eeg_raw = eeg_raw.T
eeg_filt = np.load(paths["eeg_filtered"]).squeeze()
eeg_filt = eeg_filt.reshape(-1, eeg_filt.shape[-1])
eeg_time_ms = (np.arange(eeg_raw.shape[0]) / eeg_fs) * 1000
eeg_time_sec = eeg_time_ms / 1000

# === Load fNIRS ===
fnirs_fs = 10
fnirs_raw_mat = sio.loadmat(paths["fnirs_raw"])["cnt_nback"][0, 0]
oxy_raw = fnirs_raw_mat["oxy"]
deoxy_raw = fnirs_raw_mat["deoxy"]
if isinstance(oxy_raw, np.ndarray) and oxy_raw.dtype.names:
    oxy_raw = oxy_raw[0, 0]["x"]
    deoxy_raw = deoxy_raw[0, 0]["x"]
if oxy_raw.shape[0] < oxy_raw.shape[1]:
    oxy_raw = oxy_raw.T
    deoxy_raw = deoxy_raw.T

fnirs_filt = np.load(paths["fnirs_filtered"])
fnirs_time = np.arange(fnirs_filt.shape[0]) / fnirs_fs
n_channels = fnirs_filt.shape[1] // 2
oxy_filt = fnirs_filt[:, :n_channels]
deoxy_filt = fnirs_filt[:, n_channels:]

# === Plot Setup ===
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
plt.subplots_adjust(bottom=0.3)
offset = 200

# EEG Initial window
start_idx = 0
end_idx = start_idx + eeg_window_samples

raw_plot_lines = []
filt_plot_lines = []
for i in range(eeg_raw.shape[1]):
    raw_plot_lines.append(axs[0, 0].plot(eeg_time_ms[start_idx:end_idx],
                                        eeg_raw[start_idx:end_idx, i] + i * offset, lw=0.5)[0])
    filt_plot_lines.append(axs[0, 1].plot(eeg_time_ms[start_idx:end_idx],
                                         eeg_filt[start_idx:end_idx, i] + i * offset, lw=0.5)[0])
axs[0, 0].set_title("Raw EEG")
axs[0, 1].set_title("Filtered EEG")
for ax in axs[0]:
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("EEG Channels")
    ax.set_yticks(np.arange(eeg_raw.shape[1]) * offset)
    ax.set_yticklabels([f"Ch {i+1}" for i in range(eeg_raw.shape[1])])
    ax.set_ylim(-offset, (eeg_raw.shape[1] + 1) * offset)

# fNIRS lines (initial plot only first 6 channels)
subset = list(range(6))
fnirs_colors = plt.cm.tab10(np.linspace(0, 1, 10))
raw_oxy_lines, raw_deoxy_lines = [], []
filt_oxy_lines, filt_deoxy_lines = [], []
for i in subset:
    raw_oxy_lines.append(axs[1, 0].plot(fnirs_time, oxy_raw[:, i], color=fnirs_colors[i % 10], lw=1.2)[0])
    raw_deoxy_lines.append(axs[1, 0].plot(fnirs_time, deoxy_raw[:, i], linestyle='dotted', color=fnirs_colors[i % 10], lw=1.2)[0])
    filt_oxy_lines.append(axs[1, 1].plot(fnirs_time, oxy_filt[:, i], color=fnirs_colors[i % 10], lw=1.2)[0])
    filt_deoxy_lines.append(axs[1, 1].plot(fnirs_time, deoxy_filt[:, i], linestyle='dotted', color=fnirs_colors[i % 10], lw=1.2)[0])
axs[1, 0].set_title("Raw fNIRS")
axs[1, 1].set_title("Filtered fNIRS")
for ax in axs[1]:
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Δ[Hb] (mol/L)")
    ax.set_yticks(np.linspace(-0.04, 0.4, 10))

# === Slider ===
ax_slider = plt.axes([0.2, 0.18, 0.6, 0.03])
slider = Slider(ax_slider, 'Time (s)', 0, max(0, eeg_time_sec[-1] - eeg_window_sec), valinit=0, valstep=0.1)

# === Radio Buttons for Channel Group Selection ===
channel_groups = [list(range(i, i+6)) for i in range(0, n_channels, 6)]
radio_ax = plt.axes([0.01, 0.5, 0.12, 0.35])
radio = RadioButtons(radio_ax, [f"Ch {g[0]+1}-{g[-1]+1}" for g in channel_groups])

# === Update fNIRS Channels ===
def update_channels(label):
    idx = [l.get_text() for l in radio.labels].index(label)
    global subset
    subset = channel_groups[idx]
    for i in range(len(subset)):
        raw_oxy_lines[i].set_ydata(oxy_raw[:, subset[i]])
        raw_deoxy_lines[i].set_ydata(deoxy_raw[:, subset[i]])
        filt_oxy_lines[i].set_ydata(oxy_filt[:, subset[i]])
        filt_deoxy_lines[i].set_ydata(deoxy_filt[:, subset[i]])
    update(slider.val)
    fig.canvas.draw_idle()
radio.on_clicked(update_channels)

# === Update Function ===
def update(val):
    start = int(val * eeg_fs)
    end = start + eeg_window_samples
    if end > eeg_raw.shape[0]:
        end = eeg_raw.shape[0]
        start = end - eeg_window_samples
        if start < 0: start = 0
    t_ms = eeg_time_ms[start:end]
    for i in range(len(raw_plot_lines)):
        raw_plot_lines[i].set_data(t_ms, eeg_raw[start:end, i] + i * offset)
        filt_plot_lines[i].set_data(t_ms, eeg_filt[start:end, i] + i * offset)
    axs[0, 0].set_xlim(t_ms[0], t_ms[-1])
    axs[0, 1].set_xlim(t_ms[0], t_ms[-1])

    fnirs_start = int(val * fnirs_fs)
    fnirs_end = fnirs_start + int(eeg_window_samples * fnirs_fs / eeg_fs)
    if fnirs_end > fnirs_filt.shape[0]:
        fnirs_end = fnirs_filt.shape[0]
        fnirs_start = fnirs_end - (fnirs_end - fnirs_start)
    t_fnirs = fnirs_time[fnirs_start:fnirs_end]

    min_val, max_val = np.inf, -np.inf
    for i in range(len(subset)):
        oxy_raw_seg = oxy_raw[fnirs_start:fnirs_end, subset[i]]
        deoxy_raw_seg = deoxy_raw[fnirs_start:fnirs_end, subset[i]]
        oxy_filt_seg = oxy_filt[fnirs_start:fnirs_end, subset[i]]
        deoxy_filt_seg = deoxy_filt[fnirs_start:fnirs_end, subset[i]]

        raw_oxy_lines[i].set_data(t_fnirs, oxy_raw_seg)
        raw_deoxy_lines[i].set_data(t_fnirs, deoxy_raw_seg)
        filt_oxy_lines[i].set_data(t_fnirs, oxy_filt_seg)
        filt_deoxy_lines[i].set_data(t_fnirs, deoxy_filt_seg)

        min_val = min(min_val, oxy_raw_seg.min(), deoxy_raw_seg.min(), oxy_filt_seg.min(), deoxy_filt_seg.min())
        max_val = max(max_val, oxy_raw_seg.max(), deoxy_raw_seg.max(), oxy_filt_seg.max(), deoxy_filt_seg.max())

    axs[1, 0].set_xlim(t_fnirs[0], t_fnirs[-1])
    axs[1, 1].set_xlim(t_fnirs[0], t_fnirs[-1])
    axs[1, 0].set_ylim(min_val, max_val)
    axs[1, 1].set_ylim(min_val, max_val)
    axs[1, 0].set_yticklabels([f"{y:.2e}" for y in np.linspace(min_val, max_val, len(axs[1, 0].get_yticks()))])
    axs[1, 1].set_yticklabels([f"{y:.2e}" for y in np.linspace(min_val, max_val, len(axs[1, 1].get_yticks()))])
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.tight_layout()
plt.show()
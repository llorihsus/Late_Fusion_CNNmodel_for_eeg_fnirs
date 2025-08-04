import scipy.io as sio
import numpy as np
import mne
import os
import matplotlib.pyplot as plt

def load_mat_eeg(file_path, key='eeg'):
    mat = sio.loadmat(file_path)
    data = mat[key]
    if data.shape[0] < data.shape[1]:
        data = data.T
    return data

def load_marker_file(marker_path):
    mat = sio.loadmat(marker_path)
    pos = mat['mrk']['pos'][0][0].squeeze()
    y = mat['mrk']['y'][0][0].squeeze()
    return pos, y

def preprocess_eeg(raw_data, sfreq=500, target_freq=100, l_freq=1.0, h_freq=40.0, notch=60.0):
    n_channels = raw_data.shape[1]
    info = mne.create_info(
        ch_names=[f"EEG{i}" for i in range(n_channels)],
        sfreq=sfreq,
        ch_types=['eeg'] * n_channels
    )
    raw = mne.io.RawArray(raw_data.T, info)
    raw.set_montage(mne.channels.make_standard_montage('standard_1020'))
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    raw.notch_filter(freqs=notch, verbose=False)
    if sfreq != target_freq:
        raw.resample(target_freq, verbose=False)
    return raw

def window_eeg(eeg_data, marker_pos, labels, window_size=500):
    X, y = [], []
    for pos, label in zip(marker_pos, labels):
        if pos + window_size <= eeg_data.shape[0]:
            X.append(eeg_data[pos:pos+window_size, :])
            y.append(label)
    return np.array(X), np.array(y)

def save_processed(X, y, subject_id, out_dir='processed-data/eeg'):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f'{subject_id}_eeg.npy'), X)
    np.save(os.path.join(out_dir, f'{subject_id}_labels.npy'), y)

def plot_summary(raw, subject_id):
    raw.plot(n_channels=32, title=f"Raw EEG - {subject_id}", block=False)
    psd_fig = raw.compute_psd().plot_topomap(ch_type='eeg', normalize=True, show=False)
    psd_fig[0].suptitle(f"EEG PSD Topomap - {subject_id}")
    plt.show()

# ==== MAIN EXECUTION ====

if __name__ == "__main__":
    subject_id = 's01'
    raw_dir = 'raw-data/eeg'
    out_dir = 'processed-data/eeg'
    plot = True  # Toggle this to False to disable plotting

    eeg_path = os.path.join(raw_dir, f"{subject_id}_eeg.mat")
    marker_path = os.path.join(raw_dir, f"{subject_id}_mrk_nback.mat")

    raw_eeg = load_mat_eeg(eeg_path, key='eeg')
    marker_pos, labels = load_marker_file(marker_path)

    raw_obj = preprocess_eeg(raw_eeg, sfreq=500, target_freq=100)
    if plot:
        plot_summary(raw_obj, subject_id)

    # Scale marker positions if sampling rate changed
    downsample_factor = 500 // 100
    marker_pos = marker_pos // downsample_factor

    clean_data = raw_obj.get_data().T  # shape: (samples, channels)
    X_eeg, y = window_eeg(clean_data, marker_pos, labels, window_size=500)
    save_processed(X_eeg, y, subject_id, out_dir)

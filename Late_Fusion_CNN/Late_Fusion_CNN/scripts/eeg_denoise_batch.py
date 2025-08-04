import scipy.io as sio
import numpy as np
import mne
import os

def load_mat_eeg(file_path):
    mat = sio.loadmat(file_path)
    data = mat['cnt_nback'][0, 0]['x']  # Get EEG matrix
    if data.shape[0] < data.shape[1]:
        data = data.T
    return data

def load_marker_file(marker_path):
    mat = sio.loadmat(marker_path)
    mrk = mat['mrk_nback'][0, 0]
    pos = mrk['time'].squeeze()
    y = np.argmax(mrk['y'], axis=0)  # Convert one-hot to labels
    return pos, y

def preprocess_eeg(raw_data, sfreq=200, target_freq=200, l_freq=1.0, h_freq=40.0, notch=60.0):
    n_channels = raw_data.shape[1]
    info = mne.create_info(ch_names=[f"EEG{i}" for i in range(n_channels)],
                           sfreq=sfreq, ch_types=['eeg'] * n_channels)
    raw = mne.io.RawArray(raw_data.T, info)
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    raw.notch_filter(freqs=notch, verbose=False)
    if sfreq != target_freq:
        raw.resample(target_freq, verbose=False)
    return raw.get_data().T  # shape: (samples, channels)

def window_eeg(eeg_data, marker_pos, labels, window_size=200):
    """
    eeg_data: (samples, channels)
    marker_pos: positions in samples (should already match downsampled rate)
    labels: corresponding task labels
    """
    X, y = [], []
    for pos, label in zip(marker_pos, labels):
        if pos + window_size <= len(eeg_data):
            segment = eeg_data[pos:pos+window_size, :]
            X.append(segment)
            y.append(label)
    return np.array(X), np.array(y)

def save_processed(X, y, subject_id, output_dir='processed-data/eeg'):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f'{subject_id}_eeg.npy'), X)
    np.save(os.path.join(output_dir, f'{subject_id}_labels.npy'), y)

def load_processed():
    input_eeg = [batches, fnirs_channels, samples, 1]
    input_fnirs = [batches, eeg_channels, samples, 1]
    return [input_eeg, input_fnirs]

# main
if __name__ == "__main__":
    subject_ids = [f"s{str(i).zfill(2)}" for i in range(1, 27)]  # s01 to s26
    raw_dir = 'code/ml-project/Late_Fusion_CNN/raw_data/eeg'
    out_dir = 'code/ml-project/Late_Fusion_CNN/processed_data/eeg'

    for subject_id in subject_ids:
        try:
            print(f"Processing {subject_id}...")

            eeg_path = os.path.join(raw_dir, f"{subject_id}_eeg.mat")
            marker_path = os.path.join(raw_dir, f"{subject_id}_mrk.mat")

            raw_eeg = load_mat_eeg(eeg_path)
            marker_pos, labels = load_marker_file(marker_path)

            clean_eeg = preprocess_eeg(raw_eeg, sfreq=200, target_freq=200)

            X_eeg, y = window_eeg(clean_eeg, marker_pos, labels, window_size=1000)

            X_eeg = X_eeg[..., np.newaxis]  # For CNN input shape

            save_processed(X_eeg, y, subject_id, out_dir)

        except Exception as e:
            print(f"Failed to process {subject_id}: {e}")

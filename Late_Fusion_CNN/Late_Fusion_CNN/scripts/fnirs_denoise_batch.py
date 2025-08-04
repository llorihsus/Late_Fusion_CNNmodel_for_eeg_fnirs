import scipy.io as sio
import numpy as np
import os
from scipy.signal import butter, filtfilt

def load_fnirs_data(mat_path):
    """Load fNIRS signal (samples x channels)"""
    mat = sio.loadmat(mat_path)

    # Pull top-level structure
    cnt_nback = mat['cnt_nback'][0, 0]  # MATLAB struct
    oxy_struct = cnt_nback['oxy'][0, 0]  # MATLAB object (structured array)

    # The actual signal is in the 'x' field of the struct
    oxy_data = oxy_struct['x']

    if not isinstance(oxy_data, np.ndarray):
        raise ValueError("Loaded fNIRS data is not a valid numpy array")

    # Ensure shape is (samples, channels)
    if oxy_data.shape[0] < oxy_data.shape[1]:
        oxy_data = oxy_data.T

    print(f"[DEBUG] Final oxy shape: {oxy_data.shape}")
    return oxy_data

def load_marker_file(marker_path):
    """Load marker positions and labels, downsampled to fNIRS rate"""
    mat = sio.loadmat(marker_path)
    mrk = mat['mrk_nback'][0, 0]

    # Downsample from EEG 1000 Hz to fNIRS 10 Hz
    pos_eeg = mrk['time'].squeeze()  # EEG-based marker positions
    fs_eeg = 1000
    fs_fnirs = 10
    pos = (pos_eeg * fs_fnirs // fs_eeg).astype(int)  # Downsampled for fNIRS

    y_onehot = mrk['y']
    labels = np.argmax(y_onehot, axis=0)
    return pos, labels

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, fs=10, lowcut=0.01, highcut=0.2):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data, axis=0)

def window_fnirs(data, marker_pos, labels, window_size=50):
    """Extract windows from filtered fNIRS data (5s @ 10 Hz)"""
    X, y = [], []
    for pos, label in zip(marker_pos, labels):
        if pos + window_size <= len(data):
            segment = data[pos:pos+window_size, :]
            X.append(segment)
            y.append(label)
    return np.array(X), np.array(y)

def save_processed(X, y, subject_id, out_dir='code/ml-project/Late_Fusion_CNN/processed_data/fnirs'):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f'{subject_id}_fnirs.npy'), X)
    np.save(os.path.join(out_dir, f'{subject_id}_labels.npy'), y)

# === AUTO-BATCH PREPROCESSING ===
if __name__ == "__main__":
    raw_dir = 'code/ml-project/Late_Fusion_CNN/raw_data/fnirs'
    out_dir = 'code/ml-project/Late_Fusion_CNN/processed_data/fnirs'
    subject_ids = [f's{str(i).zfill(2)}' for i in range(1, 27)]

    for subject_id in subject_ids:
        try:
            print(f'Processing {subject_id}...')

            fnirs_path = os.path.join(raw_dir, f'{subject_id}_fnirs.mat')
            marker_path = os.path.join(raw_dir, f'{subject_id}_mrk.mat')  

            if not os.path.exists(fnirs_path) or not os.path.exists(marker_path):
                print(f'Missing data for {subject_id}, skipping.')
                continue

            raw_fnirs = load_fnirs_data(fnirs_path)
            marker_pos, labels = load_marker_file(marker_path)
            filtered = bandpass_filter(raw_fnirs, fs=10)

            X, y = window_fnirs(filtered, marker_pos, labels, window_size=50)
            save_processed(X, y, subject_id, out_dir)

        except Exception as e:
            print(f'Failed to process {subject_id}: {e}')

    print("All subjects processed.")

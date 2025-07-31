import scipy.io as sio
import numpy as np
import os
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

def load_fnirs_data(mat_path, key='fnirs'):
    mat = sio.loadmat(mat_path)
    data = mat[key]
    if data.shape[0] < data.shape[1]:
        data = data.T
    return data

def load_marker_file(marker_path):
    mat = sio.loadmat(marker_path)
    pos = mat['mrk']['pos'][0][0].squeeze()
    y = mat['mrk']['y'][0][0].squeeze()
    return pos, y

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, fs=10, lowcut=0.01, highcut=0.2):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data, axis=0)

def plot_comparison(raw, filtered, subject_id):
    plt.figure(figsize=(12, 6))
    for ch in range(min(4, raw.shape[1])):
        plt.subplot(2, 2, ch + 1)
        plt.plot(raw[:, ch], label='Raw', alpha=0.6)
        plt.plot(filtered[:, ch], label='Filtered', linewidth=1.2)
        plt.title(f'Channel {ch}')
        plt.legend()
    plt.suptitle(f'fNIRS Raw vs Filtered - {subject_id}')
    plt.tight_layout()
    plt.show()

def window_fnirs(data, marker_pos, labels, window_size=100):
    X, y = [], []
    for pos, label in zip(marker_pos, labels):
        if pos + window_size <= len(data):
            segment = data[pos:pos + window_size, :]
            X.append(segment)
            y.append(label)
    return np.array(X), np.array(y)

def extract_features_fnirs(X):
    """
    X: np.array of shape (n_windows, window_size, n_channels)
    Returns: np.array of shape (n_windows, n_channels × 5)
    Features: mean, std, peak, time_to_peak, AUC
    """
   
    n_windows, win_len, n_channels = X.shape
    features = []

    for window in X:
        feats = []
        for ch in range(n_channels):
            signal = window[:, ch]
            mean = np.mean(signal)
            std = np.std(signal)
            peak = np.max(signal)
            ttp = np.argmax(signal)  # time to peak (sample index)
            auc = np.trapz(signal)   # area under curve
            feats.extend([mean, std, peak, ttp, auc])
        features.append(feats)

    return np.array(features)


def save_processed(X, y, subject_id, out_dir='processed-data/fnirs'):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f'{subject_id}_fnirs.npy'), X)
    np.save(os.path.join(out_dir, f'{subject_id}_labels.npy'), y)

# === Run Single Subject ===
if __name__ == "__main__":
    subject_id = 's01'
    plot = True  # Set to False to skip plotting

    fnirs_path = f'raw-data/fnirs/{subject_id}_fnirs.mat'
    marker_path = f'raw-data/fnirs/{subject_id}_mrk_nback.mat'
    output_dir = 'processed-data/fnirs'

    raw_fnirs = load_fnirs_data(fnirs_path)
    marker_pos, labels = load_marker_file(marker_path)
    filtered_fnirs = bandpass_filter(raw_fnirs, fs=10)

    if plot:
        plot_comparison(raw_fnirs, filtered_fnirs, subject_id)

    X_fnirs, y = window_fnirs(filtered_fnirs, marker_pos, labels, window_size=100)
    save_processed(X_fnirs, y, subject_id, output_dir)

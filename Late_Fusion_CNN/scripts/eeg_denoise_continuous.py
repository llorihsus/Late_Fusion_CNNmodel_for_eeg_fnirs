import scipy.io as sio
import numpy as np
import mne
import os

def load_mat_eeg(file_path):
    mat = sio.loadmat(file_path)
    data = mat['cnt_nback'][0, 0]['x']
    if data.shape[0] < data.shape[1]:
        data = data.T
    return data

def preprocess_eeg(raw_data, sfreq=1000, target_freq=500, l_freq=1.0, h_freq=40.0, notch=60.0):
    # Clean up bad values
    raw_data = np.nan_to_num(raw_data)  # Replace NaNs/Infs with 0
    raw_data -= np.mean(raw_data, axis=0)  # Demean per channel

    n_channels = raw_data.shape[1]
    info = mne.create_info(ch_names=[f"EEG{i}" for i in range(n_channels)],
                           sfreq=sfreq, ch_types=['eeg'] * n_channels)
    raw = mne.io.RawArray(raw_data.T, info)
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    raw.notch_filter(freqs=notch, verbose=False)
    if sfreq != target_freq:
        raw.resample(target_freq, verbose=False)
    return raw.get_data().T  # Final shape: (samples, channels)

def save_continuous_eeg(data, subject_id, output_dir='code/ml-project/Late_Fusion_CNN/processed_data/eeg/continuous'):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f'{subject_id}_eeg_cnt.npy'), data)

# === Main: Apply to All Subjects ===
if __name__ == "__main__":
    subject_ids = [f"s{str(i).zfill(2)}" for i in range(1, 27)]
    raw_dir = 'code/ml-project/Late_Fusion_CNN/raw_data/eeg'
    out_dir = 'code/ml-project/Late_Fusion_CNN/processed_data/eeg/continuous'

    for subject_id in subject_ids:
        try:
            print(f"Processing {subject_id}...")
            eeg_path = os.path.join(raw_dir, f"{subject_id}_eeg.mat")
            raw_eeg = load_mat_eeg(eeg_path)
            filtered_eeg = preprocess_eeg(raw_eeg, sfreq=1000, target_freq=500)
            save_continuous_eeg(filtered_eeg, subject_id, out_dir)
        except Exception as e:
            print(f"Failed to process {subject_id}: {e}")
        
    print("All subjects processed.")

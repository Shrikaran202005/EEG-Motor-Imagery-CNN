import os
import numpy as np
import mne

def preprocess_gdf_folder(folder_path, fs=250, trial_window_sec=4):
    """
    Preprocess GDF files from a folder OR a single file.
    Returns: X (data), y (labels), subject_ids
    """

    X, y, subject_ids = [], [], []

    # ✅ Handle single file vs folder
    if folder_path.endswith(".gdf") and os.path.isfile(folder_path):
        filenames = [os.path.basename(folder_path)]
        folder_dir = os.path.dirname(folder_path)
    else:
        filenames = sorted(os.listdir(folder_path))
        folder_dir = folder_path

    for filename in filenames:
        if not filename.endswith(".gdf"):
            continue

        filepath = os.path.join(folder_dir, filename)
        print(f"Processing: {filepath}")

        try:
            # Load GDF file with MNE
            raw = mne.io.read_raw_gdf(filepath, preload=True, verbose="ERROR")
            events, event_id = mne.events_from_annotations(raw, verbose="ERROR")

            # Pick EEG channels
            raw.pick_types(eeg=True)

            # Epoching
            tmin, tmax = 0, trial_window_sec  # seconds
            epochs = mne.Epochs(
                raw, events, event_id=None, tmin=tmin, tmax=tmax,
                baseline=None, preload=True, detrend=1, verbose="ERROR"
            )

            data = epochs.get_data()  # shape: (trials, channels, samples)

            # Labels
            labels = epochs.events[:, -1]

            # Store results
            X.append(data.astype(np.float32))
            y.append(labels.astype(np.int64))
            subject_ids.extend([filename] * len(labels))

        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")

    if len(X) == 0:
        print("❌ No valid data found.")
        return np.array([]), np.array([]), np.array([])

    # Concatenate all subjects
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    subject_ids = np.array(subject_ids)

    print(f"✅ Done. Generated {X.shape[0]} samples from {len(set(subject_ids))} subject(s).")
    print(f"🟢 X shape: {X.shape}, y shape: {y.shape}, subject_ids shape: {subject_ids.shape}")

    return X, y, subject_ids

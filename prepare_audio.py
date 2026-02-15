import librosa
import numpy as np
import pandas as pd
import os

df = pd.read_csv("dataset.csv")
os.makedirs("audio_features", exist_ok=True)

def extract_features(path):
    y, sr = librosa.load(path, sr=22050)

    # trim silence
    y, _ = librosa.effects.trim(y)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # pitch
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = pitches[magnitudes > np.median(magnitudes)]

    if len(pitch) == 0:
        pitch = np.array([0])

    pitch_mean = np.mean(pitch)
    pitch_std = np.std(pitch)

    # energy
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    # spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # zero crossing
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        pitch_mean, pitch_std,
        rms_mean, rms_std,
        np.mean(contrast, axis=1),
        zcr_mean
    ])

    return features

for i, row in df.iterrows():
    feat = extract_features(row["path"])
    np.save(f"audio_features/{i}.npy", feat)

    if i % 200 == 0:
        print("Processed", i)

print("All audio features saved")

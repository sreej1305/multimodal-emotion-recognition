import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load dataset
df = pd.read_csv("dataset.csv")

emotions = sorted(df["emotion"].unique())
label_map = {e:i for i,e in enumerate(emotions)}
df["label"] = df["emotion"].map(label_map)

MAX_LEN = 100

def extract_features(path):
    y, sr = librosa.load(path, sr=16000)
    y, _ = librosa.effects.trim(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T

    if len(mfcc) < MAX_LEN:
        pad_width = MAX_LEN - len(mfcc)
        mfcc = np.pad(mfcc, ((0,pad_width),(0,0)), mode='constant')
    else:
        mfcc = mfcc[:MAX_LEN]

    return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

# split same as training
_, test_df = train_test_split(df, test_size=0.2, stratify=df["label"])

class LSTMModel(nn.Module):
    def __init__(self, input_size=40, hidden=128, classes=len(emotions)):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

model = LSTMModel()
model.load_state_dict(torch.load("speech_model.pt"))
model.eval()

preds, trues = [], []

for _, row in test_df.iterrows():
    x = extract_features(row["path"])
    with torch.no_grad():
        out = model(x)
    pred = torch.argmax(out).item()

    preds.append(pred)
    trues.append(row["label"])

acc = accuracy_score(trues, preds)
print("Speech Model Accuracy:", acc)

import torch, torch.nn as nn, numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("../fusion_pipeline/dataset.csv")

emotions = sorted(df["emotion"].unique())
label_map = {e:i for i,e in enumerate(emotions)}
df["label"] = df["emotion"].map(label_map)

# =========================
# DATASET
# =========================
class SpeechDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        real_index = row.name

        audio = np.load(f"../fusion_pipeline/audio_features/{real_index}.npy")  # (52,)
        label = row["label"]

        return torch.tensor(audio, dtype=torch.float32), torch.tensor(label)

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

train_loader = DataLoader(SpeechDataset(train_df), batch_size=64, shuffle=True)
test_loader  = DataLoader(SpeechDataset(test_df), batch_size=64)

# =========================
# MODEL (MLP for vector features)
# =========================
class SpeechModel(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(52,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,classes)
        )

    def forward(self,x):
        return self.net(x)

model = SpeechModel(len(emotions))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================
# TRAIN
# =========================
for epoch in range(10):
    total_loss = 0
    for x,y in train_loader:
        out = model(x)
        loss = criterion(out,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch",epoch+1,"Loss:",total_loss)

torch.save(model.state_dict(),"speech_model.pt")
print("Model saved")

# =========================
# TEST ACCURACY
# =========================
correct=total=0
with torch.no_grad():
    for x,y in test_loader:
        pred=model(x).argmax(1)
        correct+=(pred==y).sum().item()
        total+=len(y)

print("Speech Accuracy:",100*correct/total)

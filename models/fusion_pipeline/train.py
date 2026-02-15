import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# ---------------- LOAD DATA ----------------
df = pd.read_csv("dataset.csv")

df = df.dropna(subset=["emotion"])
df["emotion"] = df["emotion"].str.lower().str.strip()

# merge duplicate label
df["emotion"] = df["emotion"].replace({
    "surprised": "surprise",
    "pleasant_surprise": "surprise"
})

df = df.reset_index(drop=True)

print("Dataset distribution:\n", df["emotion"].value_counts())

# label encoding
emotions = sorted(df["emotion"].unique())
label_map = {e:i for i,e in enumerate(emotions)}
df["label"] = df["emotion"].map(label_map)

# ---------------- DATASET ----------------
# ---------------- DATASET ----------------
class FusionDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe   # DO NOT reset index

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        real_index = row.name   # original dataset index

        audio = np.load(f"audio_features/{real_index}.npy")
        text  = np.load(f"text_features/{real_index}.npy")

        audio = torch.tensor(audio, dtype=torch.float32)
        text  = torch.tensor(text, dtype=torch.float32)
        label = torch.tensor(row["label"], dtype=torch.long)

        return audio, text, label


# ---------------- SPLIT ----------------
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

train_loader = DataLoader(FusionDataset(train_df), batch_size=64, shuffle=True)
test_loader  = DataLoader(FusionDataset(test_df), batch_size=64)
# ---------------- MODEL ----------------
class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # audio branch
        self.audio_net = nn.Sequential(
            nn.Linear(52, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        # text branch
        self.text_net = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, audio, text):
        a = self.audio_net(audio)
        t = self.text_net(text)
        x = torch.cat((a, t), dim=1)
        return self.classifier(x)

model = FusionModel(len(emotions))

# ---------------- TRAIN ----------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 15

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for audio, text, label in train_loader:
        output = model(audio, text)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}  Loss: {total_loss:.4f}")

# save
torch.save({
    "model": model.state_dict(),
    "emotions": emotions
}, "fusion_model.pt")

print("\nTraining Finished & Model Saved")

# ---------------- TEST ACCURACY ----------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for audio, text, label in test_loader:
        output = model(audio, text)
        pred = torch.argmax(output, dim=1)

        correct += (pred == label).sum().item()
        total += label.size(0)

accuracy = 100 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")

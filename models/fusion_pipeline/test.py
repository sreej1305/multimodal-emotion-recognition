import torch
import numpy as np
import os
import torch.nn as nn

# load checkpoint
checkpoint = torch.load("fusion_model.pt")
emotions = checkpoint["emotions"]

# model
class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(40 + 768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, audio, text):
        x = torch.cat((audio, text), dim=1)
        return self.net(x)

model = FusionModel(len(emotions))
model.load_state_dict(checkpoint["model"])
model.eval()

# load samples
texts = []
audios = []

for i in range(20):
    text = np.load(f"text_features/{i}.npy")
    audio = np.load(f"audio_features/{i}.npy")

    audio = audio.mean(axis=0)

    texts.append(text)
    audios.append(audio)

text = torch.tensor(np.array(texts), dtype=torch.float32)
audio = torch.tensor(np.array(audios), dtype=torch.float32)

with torch.no_grad():
    outputs = model(audio, text)

pred = torch.argmax(outputs, dim=1)

print("\nFirst 20 predictions:\n")
for i in range(20):
    print(i, "->", emotions[pred[i].item()])

import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import os

df = pd.read_csv("dataset.csv")

SAVE_DIR = "text_features"
os.makedirs(SAVE_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
model.eval()

for i,row in df.iterrows():
    tokens = tokenizer(row["text"], return_tensors="pt", padding="max_length", truncation=True)

    with torch.no_grad():
        output = model(**tokens)

    embedding = output.last_hidden_state[:,0,:].squeeze(0).numpy()
    np.save(f"{SAVE_DIR}/{i}.npy", embedding)

    if i % 200 == 0:
        print("Processed", i)

print("All text features saved")

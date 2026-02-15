import os
import pandas as pd

DATASET_DIR = "data"

rows = []

for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        if file.endswith(".wav"):

            emotion = os.path.basename(root).lower()

            # FIX DUPLICATE LABEL
            if emotion == "surprised":
                emotion = "surprise"

            rows.append({
                "path": os.path.join(root, file),
                "emotion": emotion,
                "text": file.split("_")[-1].replace(".wav","")
            })

df = pd.DataFrame(rows)
df.to_csv("models/fusion_pipeline/dataset.csv", index=False)

print("\nDataset rebuilt with emotions:")
print(df["emotion"].value_counts())

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load dataset
df = pd.read_csv("dataset.csv")
df = df[["text","emotion"]]

labels = sorted(df["emotion"].unique())
label_map = {e:i for i,e in enumerate(labels)}
df["label"] = df["emotion"].map(label_map)

# same split as training
_, test_df = train_test_split(df, test_size=0.2, stratify=df["label"])

dataset = Dataset.from_pandas(test_df[["text","label"]])

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("text_model")
model.eval()

preds, trues = [], []

for item in dataset:
    inputs = {
        "input_ids": torch.tensor(item["input_ids"]).unsqueeze(0),
        "attention_mask": torch.tensor(item["attention_mask"]).unsqueeze(0),
    }

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits).item()
    preds.append(pred)
    trues.append(item["label"])

acc = accuracy_score(trues, preds)
print("Text Model Accuracy:", acc)

# Multimodal Emotion Recognition

This project implements a multimodal emotion recognition system using Speech, Text, and Fusion pipelines. It integrates audio and text features to classify emotions.

## Project Structure

```
multimodal-emotion-recognition/
│
├── models/
│   ├── speech_pipeline/     # Speech emotion recognition model
│   │   ├── train.py
│   │   └── test.py
│   ├── text_pipeline/       # Text emotion recognition model
│   │   ├── train.py
│   │   └── test.py
│   └── fusion_pipeline/     # Multimodal fusion model (Speech + Text)
│       ├── train.py
│       └── test.py
│
├── Results/
│   └── plots/
│       └── model_comparison.png  # Performance comparison plot
│
├── dataset.csv              # Dataset containing emotion labels and metadata
├── requirements.txt         # Python dependencies
├── report.pdf               # Detailed project report
└── README.md                # Project documentation
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sreej1305/multimodal-emotion-recognition.git
    cd multimodal-emotion-recognition
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The project consists of three main pipelines. You can train and test each pipeline independently.

### 1. Speech Pipeline
Trains a model using audio features.
```bash
cd models/speech_pipeline
python train.py
python test.py
```

### 2. Text Pipeline
Trains a model using text features.
```bash
cd models/text_pipeline
python train.py
python test.py
```

### 3. Fusion Pipeline
Trains a fusion model that combines both speech and text features for improved accuracy.
```bash
cd models/fusion_pipeline
python train.py
python test.py
```

## Results
The comparison of model performance can be found in `Results/plots/model_comparison.png`.
Error analysis and detailed methodology are available in `report.pdf`.

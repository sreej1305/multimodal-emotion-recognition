# Multimodal Emotion Recognition
This project compares Speech, Text and Multimodal emotion recognition models using the TESS dataset.
## Results

| Model | Accuracy |
|------|------|
| Speech Only | 90.71% |
| Text Only | 13.57% |
| Fusion | 67.86% |

Speech carries emotion while text remains same across emotions in TESS dataset.

## Run
Speech:
cd models/speech_pipeline
python train.py
Text:
cd models/text_pipeline
python train.py
Fusion:
cd models/fusion_pipeline
python train.py
python predict.py

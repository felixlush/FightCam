# FightCam - Realtime violence detection in CCTV Footage

Fightcam is a deep-learning based computer vision project that detects violent activity in video footage using a CNN-LSTM architecture.  FightCam is Built on a CNN-LSTM architecture. The model uses spatial feature extraction using a pretrain ResNet50 in combination with temporal modeling from a BiLSTM to classify videos as either "Violent" or "Non Violent". 

Trained on the RWF-2000 dataset, the model achieves an accuracy of 86.67%, demonstrating strong performance across a range of environmental conditions. Through ablation studies and comparative analysis with state-of-the-art methods, FightCam balances computational efficiency and predictive power, making it suitable for real-world deployment on standard surveillance infrastructure.

This project was part of a computer science capstone as is presented as a complete self contained pipeline in a single Juypter Notebook

## What does it do?
- Processes video input by sampling frames
- Extracts spatial features using a pretrained ResNet50
- Learns temporal patterns using a BiLSTM
- Outputs a binary classification: 0 -> Violent, 1 -> No Violence

<div style="display: flex;">
  <img width="390" height="410" alt="Screenshot 2026-04-22 at 2 41 53 pm" src="https://github.com/user-attachments/assets/2e2b31c0-c328-461c-adfe-aa9889164ae8" />
<img width="391" height="412" alt="Screenshot 2026-04-22 at 2 42 00 pm" src="https://github.com/user-attachments/assets/b2818dad-fefd-4cc2-a1b8-7d525e9146c9" />
</div>

## Results

Accuracy: 0.8667 <br>
Precision: 0.8519 <br>
Recall: 0.8734 <br>
F1 Score: 0.8625 <br>
Specificity: 0.8310 <br>


## Model Architecture

The pipeline for the model is made up of four stages

1. Video Processing
  - 20 frames are randomly sampled from each video to improve generalisation and reduce compute + training cost
2. Feature Extraction
3. Temporal Pattern Extraction
4. Classification


<img width="1426" height="951" alt="FightCam3" src="https://github.com/user-attachments/assets/29b5cf5c-d56f-402a-b671-a6327c01bd5d" />

## Dataset

## How to run



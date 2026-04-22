# FightCam - Realtime violence detection in CCTV Footage

Fightcam is a deep-learning based computer vision project that detects violent activity in video footage using a CNN-LSTM architecture.  FightCam is Built on a CNN-LSTM architecture. The model uses spatial feature extraction using a pretrain ResNet50 in combination with temporal modeling from a BiLSTM to classify videos as either "Violent" or "Non Violent". 

Trained on the RWF-2000 dataset, the model achieves an accuracy of 86.67%, demonstrating strong performance across a range of environmental conditions. Through ablation studies and comparative analysis with state-of-the-art methods, FightCam balances computational efficiency and predictive power, making it suitable for real-world deployment on standard surveillance infrastructure.

This project was part of a computer science capstone as is presented as a complete self contained pipeline in a single Juypter Notebook

## What does it do?
- Processes video input by sampling frames
- Extracts spatial features using a pretrained ResNet50
- Learns temporal patterns using a BiLSTM
- Outputs a binary classification: 0 -> Violent, 1 -> No Violence

<table>
  <tr>
    <td>
      <img width="390" height="410" alt="Screenshot 2026-04-22 at 2 41 53 pm" src="https://github.com/user-attachments/assets/2e2b31c0-c328-461c-adfe-aa9889164ae8" />
    </td>
    <td>
      <img width="391" height="412" alt="Screenshot 2026-04-22 at 2 42 00 pm" src="https://github.com/user-attachments/assets/b2818dad-fefd-4cc2-a1b8-7d525e9146c9" />
    </td>
  </tr>
</table>
  

</div>

## Results

<table>
  <tr>
    <th>Metric</th>
    <th>Score</th>
  </tr>
  <tr>
    <td>Accuracy</td>
    <td>0.8667</td>
  </tr>
  <tr>
    <td>Precision</td>
    <td>0.8554</td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>0.8987</td>
  </tr>
  <tr>
    <td>F1 Score</td>
    <td>0.8765</td>
  </tr>
</table>

**Confusion Matrix**
<img width="559" height="432" alt="FightCamMatrix" src="https://github.com/user-attachments/assets/882a7231-09bd-4695-9fa9-dd4e4f0d78f6" />


## Model Architecture

The pipeline for the model is made up of four stages

1. Video Processing
  - 20 frames are randomly sampled from each video to improve generalisation and reduce compute + training cost.
  - Frames are resized to 224x224
  - Passed through augmentation (colour jitter, random horizontal flip ect)
2. Feature Extraction
  - Each frame is encoded by a pretrained ResNet50 with the final layer removed
  - Produdes a feature vector per frame
3. Temporal Pattern Extraction
  - The 20 feature vectors from each videos are then fed as a sequence into a 2-layer BiLSTM (hidden size 256) which learning motion and escalation patterns
4. Classification
  - Final hidden layer state goes through a small MLP
  - MLP consisits of (Linear -> Relu -> Dropout -> Linear) to produce Violent/Non Violent Prediction


<img width="1426" height="951" alt="FightCamWhiteBackground" src="https://github.com/user-attachments/assets/21b75f45-bcfc-4127-8b72-716761276784" />


## Dataset
In this project we used the RWF-2000 Dataset. RWF-2000 Consists of 2000 real world surveillance clips each arounf 5 seconds long. Before training we processed the dataset by decoding each video into JPEG frames with OpenCV

## How to run
**Requirements:**
- Python 3.9+
- Pytorch
- Torchvision, numpy, pillow, scikit-learn, matplotlib, tqdm, tensorboard

**Setup**
1. Clone the repo
2. Get the dataset from RWF official <a src="https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection">Repo</a>
3. Point the notebook to the datasource via the dataset_ath variable
4. Run the notebook from top to bottom


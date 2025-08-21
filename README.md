EEG-Motor-Imagery-Classification
📌 Overview

This project implements a spectrogram-based deep learning pipeline for EEG Motor Imagery (MI) classification, leveraging the BCI Competition IV Datasets (2a & 2b). Using advanced preprocessing techniques and a custom CNN architecture enhanced with Squeeze-and-Excitation (SE) blocks, the system achieves high classification accuracy across subjects.

The pipeline transforms raw EEG signals into Short-Time Fourier Transform (STFT) spectrograms and trains robust deep learning models to recognize motor imagery tasks such as left hand, right hand, foot, and tongue movements.

✨ Features

Preprocessing pipeline:

Common Average Referencing (CAR)

Notch Filter (50 Hz)

Bandpass Filter (8–30 Hz, Mu & Beta rhythms)

Min–Max Normalization

Sliding Window Segmentation

STFT-based Spectrogram Generation

Deep Learning Architecture:

Convolutional Neural Network (CNN)

Squeeze-and-Excitation (SE) blocks for channel attention

Spectrogram Augmentation (time & frequency masking)

Lazy Dataset Loading for large EEG files

Datasets:

BCI Competition IV – Dataset 2a (4-class: left, right, foot, tongue)

BCI Competition IV – Dataset 2b (2-class: left, right)

Results:

Dataset 2a → ~90–96% accuracy

Dataset 2b → ~97–99% accuracy

📂 Repository Structure
├── data/                # EEG datasets (.gdf or preprocessed .npy)
├── preprocessing/       # Preprocessing scripts (CAR, filters, STFT, segmentation)
├── models/              # CNN architectures with SE blocks
├── training/            # Training pipeline & evaluation scripts
├── utils/               # Helper functions (augmentation, dataset loader)
├── results/             # Saved models, logs, and evaluation metrics
└── README.md

🚀 Getting Started
1. Clone Repository
git clone https://github.com/your-username/EEG-Motor-Imagery-Classification.git
cd EEG-Motor-Imagery-Classification

2. Install Dependencies
pip install -r requirements.txt

3. Dataset Setup

Download BCI Competition IV 2a/2b datasets (.gdf files).

Place them inside data/ directory.

4. Run Preprocessing
python preprocessing/preprocess_gdf.py --dataset 2a

5. Train Model
python training/train_cnn.py --dataset 2a --augment True

6. Evaluate Model
python training/evaluate.py --dataset 2a

📊 Results & Analysis

2a (4-class MI): Achieved ~90–96% accuracy, with some misclassifications between left/right and foot/tongue imagery.

2b (2-class MI): Achieved ~97–99% accuracy, with minimal confusion.

Average confusion matrices demonstrate strong diagonal dominance across subjects.

🧠 Applications

Brain-Computer Interfaces (BCIs)

Neuro-rehabilitation

Assistive technologies for motor-impaired individuals

Real-time control systems

📜 License

This project is released under the MIT License.

👤 Author

Developed as part of an academic project at NIT.

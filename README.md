# Hopenet Head Pose Estimation

This project implements **real-time head pose estimation** using the **Hopenet architecture** in PyTorch.  
It supports training, evaluation, and inference from **images**.

---

## 📌 Features
- Train Hopenet on **300W-LP** dataset with configurable loss functions (MSE, MAE).
- Evaluate on **AFLW2000** dataset with metrics & visualizations:
  - RMSE, STD of yaw/pitch/roll
  - Error distribution histograms
  - Cumulative error distribution
  - Error vs. Angle scatter plots
- Inference modes:
  - **Image Inference**: Run on single images or folders of images.
- Data preprocessing, augmentation, and cropping pipelines provided in `datasets.py`.

---

## 📂 Repository Structure
```
.
├── train_hopenet.py      # Training script
├── test_hopenet.py       # Evaluation & metrics
├── image_inference.py    # Run inference on images
├── webcam_inference.py   # Run inference on live webcam feed
├── datasets.py           # Dataset loaders (300W-LP, AFLW2000)
├── hopenet.py            # Hopenet model definition
├── util/
│   └── utils.py          # Utility functions (metrics, plotting, preprocessing)
└── README.md             # Project documentation
```

---

## 🚀 Getting Started

### 1️⃣ Install Dependencies
Create a virtual environment and install required packages:

```bash
pip install -r requirements.txt
```

### 2️⃣ Dataset Preparation
Download datasets and organize them as:
```
datasets/
├── 300W_LP/
└── AFLW2000/
```

Generate train/val/test splits:
```bash
python datasets.py
```

This will create:
- `train_filename_all.npy`
- `val_filename_all.npy`
- `dev_filename_all.npy`
- `test_filename.npy`

### 3️⃣ Training
Run training on 300W-LP:

```bash
python train_hopenet.py     --gpu 0     --num_epochs 15     --batch_size 64     --lr 0.0001     --dataset Pose_300W_LP     --data_dir ./datasets/300W_LP     --filename_list ./     --output_dir ./results     --Loss_func MSE     --class_weight balanced
```

### 4️⃣ Evaluation
Run evaluation on AFLW2000:

```bash
python test_hopenet.py     --gpu 0     --snapshot ./results/output/snapshots/model_best.pkl     --dataset AFLW2000     --data_dir ./datasets/AFLW2000     --filename_list ./     --output_dir ./results/eval
```

### 5️⃣ Inference

#### Image Inference
```bash
python image_inference.py     --snapshot ./results/output/snapshots/model_best.pkl     --image ./sample.jpg
```

---

## 📊 Visualizations
- **Error Distribution Histograms** (yaw, pitch, roll)
- **Cumulative Error Distribution** (percentage of samples below error thresholds)
- **Error vs. Angle Scatter Plots** (bias & variance analysis)

These are automatically saved in the output directory after evaluation.

---

## 📝 Requirements
To export your environment dependencies:
```bash
pip freeze > requirements.txt
```

Minimal `requirements.txt` should include:
```
torch
torchvision
opencv-python
matplotlib
scikit-learn
numpy
pandas
tqdm
Pillow
joblib
```

---

## 📌 References
- [Hopenet: Fine-Grained Head Pose Estimation Without Keypoints](https://arxiv.org/abs/1710.00925)
- Pretrained ResNet-50 backbone from [PyTorch Model Zoo](https://pytorch.org/vision/stable/models.html)

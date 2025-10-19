# Rock–Paper–Scissors Classification using CNNs

This repository contains the full implementation and report for a machine learning project.

The task was to classify hand gestures (rock, paper, scissors) using Convolutional Neural Networks (CNNs), evaluate different architectures, and apply good ML practices in training and evaluation.

---

## Repository Structure

```
├── _rps.ipynb                   # Final Jupyter Notebook with all code and results
├── ML_Project_Damir_Uvayev.pdf  # Final project report
├── outputs/
│   ├── plots/                   # All generated figures (curves, confusion matrices, misclassifications)
│   └── checkpoints/
│       └── link.txt             # Link to Google Drive folder containing model checkpoints (.pt)
├── data/
│   └── raw/                     # Dataset folders with .png images
│       ├── rock/
│       ├── paper/
│       └── scissors/
```

>  **Note:** Due to GitHub file size limits, model checkpoints (.pt) and dataset images are not included in the repository.

---

## Model Architectures

All models were custom-built Convolutional Neural Networks:

* **Model A**: Simple 2-layer CNN (baseline).
* **Model B**: 4-layer CNN with dropout — best performer.
* **Model C**: Larger 5-layer CNN with more filters — similar performance to B.

---

## Models Evaluated

| Model   | Params | Val Acc | Test Acc | Training Time |
| ------- | ------ | ------- | -------- | ------------- |
| Model A | 2.1M   | 98.5%   | 94.8%    | 2.99 min      |
| Model B | 8.4M   | 99.4%   | 97.3%    | 8.70 min      |
| Model C | 33.7M  | 98.8%   | 97.3%    | 10.07 min     |

Model B offered the best trade-off between accuracy and complexity and is recommended for deployment.

---

## Reproducibility

You can either:

* **(Recommended)** Download pretrained model checkpoints from [Download checkpoints from Google Drive](https://drive.google.com/drive/folders/12WBBe9__R3kHE_jM1in2ABQXMFUMQsTP?usp=sharing)
or copy the link manually from link.txt file in outputs/checkpoints to skip training and reproduce the final metrics **exactly**.
* **Or** train all models from scratch using the notebook (results will be close, but may vary slightly).

Steps:

1. Download the Rock–Paper–Scissors image dataset and place the folders (`rock/`, `paper/`, `scissors/`) into `data/raw/`.
2. (Optional) Download `.pt` checkpoint files from the Drive link above and place them into `outputs/checkpoints/`.
3. Run `_rps.ipynb` to load or train the models and evaluate results.

---

## Dataset

* Rock-Paper-Scissors Images. Link: https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors

---

## Environment

Developed with:

* Python 3.11.9
* PyTorch 2.8.0 (CPU)
* Torchvision 0.15+
* NumPy, Pandas, scikit-learn, Matplotlib, PIL

Install dependencies:

```bash
pip install torch torchvision pandas scikit-learn matplotlib pillow
```

# Rock–Paper–Scissors Classification using CNNs

This repository contains the full implementation and report for a machine learning project.

The task was to classify hand gestures (rock, paper, scissors) using Convolutional Neural Networks (CNNs), evaluate different architectures, and apply good ML practices in training and evaluation.

---

## Repository Structure

```
├── _rps.ipynb                  # Final Jupyter Notebook with all code and results
├── ML_Project_Damir_Uvayev.pdf  # Final project report
├── outputs/
│   └── plots/                 # All generated figures (curves, confusion matrices, misclassifications)
├── data/
│   └── raw/                   # Dataset folders with .png images
│       ├── rock/
│       ├── paper/
│       └── scissors/
```


> **Note:** Due to GitHub file size limits, model checkpoints (.pt) and dataset images are not included. See below for reproducibility.
---

### Model Architectures

All models were custom-built Convolutional Neural Networks:

* **Model A**: A basic CNN with 2 convolutional layers and a shallow classifier.
* **Model B**: A deeper CNN with 4 convolutional layers and dropout regularization — the best performer.
* **Model C**: A large CNN with 5 convolutional layers and high parameter count, but no accuracy gain over Model B.

---

## Models Evaluated
Three CNN models of increasing complexity were implemented:

| Model   | Params     | Val Acc | Test Acc | Training Time |
|---------|------------|---------|----------|----------------|
| Model A | 2.1M       | 98.5%   | 94.8%    | 2.99 min       |
| Model B | 8.4M       | 99.4%   | 97.3%    | 8.70 min       |
| Model C | 33.7M      | 98.8%   | 97.3%    | 10.07 min      |

Model B achieved the best trade-off between accuracy and complexity and is recommended for deployment.

---

## Reproducibility
- All data splits (train/val/test) are generated programmatically using a fixed `random.seed=42`
- Model weights are saved locally during training but are **not included** in this repo due to GitHub’s size limits
- To reproduce results:
  1. Download the Rock–Paper–Scissors image dataset and place the extracted folders (rock/, paper/, scissors/) inside a directory named data/raw/ in the project root.
  2. Run `_rps.ipynb` from start to finish
  3. All models will be trained and evaluated automatically

---

## Environment
This project was developed with:

- Python 3.11.9
- PyTorch 2.8.0 (CPU version)
- Torchvision 0.15+
- NumPy, Pandas, scikit-learn, Matplotlib, PIL

Install the required packages:

pip install torch torchvision pandas scikit-learn matplotlib pillow

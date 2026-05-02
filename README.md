# 🎲 Dice Face Classifier — CNN Image Classifier

A PyTorch-based convolutional neural network that classifies the number of dots on a dice face (1–6) from 28×28 grayscale images. Built as part of the DSA8021 Machine Learning coursework (2026).

---

## Overview

This project trains a CNN to perform **6-class image classification** on a custom dice image dataset. Each image is a 28×28 grayscale pixel grid, and the model predicts which face of a dice is shown (values 1–6).

| Metric | Result |
|---|---|
| Validation Accuracy | 93.51% |
| Test Accuracy | 85.42% |

---

## Project Structure
dice-face-classifier-cnn/
- DSA8021_ML_practical_aamnaarif_40503414.ipynb 
- data.csv                                      (Training/validation data)
- test.csv                                      (Held-out test data)
- best_dice_cnn_model.pth                       (Saved model weights (after training))

---

## Model Architecture

The final model, `ImprovedDiceCNN`, consists of:

- **Conv Block 1:** 1→16 filters, 3×3 kernel, BatchNorm, ReLU, MaxPool (28×28 → 14×14)
- **Conv Block 2:** 16→32 filters, 3×3 kernel, BatchNorm, ReLU, MaxPool (14×14 → 7×7)
- **Fully Connected:** 32×7×7 → 128 → 6 output logits
- **Loss Function:** CrossEntropyLoss
- **Optimiser:** Adam (lr=0.0003, weight_decay=1e-4)
- **Regularisation:** Dropout, BatchNorm, early stopping (patience=7)

---

## Training Pipeline

1. **Data Exploration** — class distribution, pixel intensity analysis, qualitative image inspection
2. **Preprocessing** — pixel normalisation to [0, 1], label encoding (1–6 → 0–5), stratified 80/20 train-val split
3. **Augmentation** — random rotations and translations applied to training data only
4. **Training** — 50 epochs max with early stopping; best checkpoint saved by validation loss
5. **Evaluation** — confusion matrix, classification report, and prediction time on test set

---

## Getting Started

### Requirements

```bash
pip install torch torchvision scikit-learn pandas numpy matplotlib
```

| Package | Version |
|---|---|
| torch | 2.6.0 |
| torchvision | 0.21.0 |
| scikit-learn | 1.6.1 |
| pandas | 2.2.3 |
| numpy | 2.2.6 |
| matplotlib | 3.10.3 |

### Run

```bash
jupyter notebook DSA8021_ML_practical_aamnaarif_40503414.ipynb
```

### Load the Saved Model

```python
import torch

model = ImprovedDiceCNN()
model.load_state_dict(torch.load("best_dice_cnn_model.pth"))
model.eval()
```

---

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---

## License

This project is licensed under the [MIT License](LICENSE).

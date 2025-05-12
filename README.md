# 🧠 Deep Learning on CIFAR-10 and MNIST


---

## 📘 Project Overview

This project explores the use of various deep learning architectures — **AlexNet**, **VGGNet**, and **ResNet** — to classify images from the **CIFAR-10** and **MNIST** datasets. The goal is to evaluate and compare the performance of these models on image classification tasks.

---

## 🗂️ Datasets Used

- **MNIST**: Handwritten digits (28x28 grayscale images, 10 classes)
- **CIFAR-10**: Color images (32x32 RGB images, 10 classes)

---

## 🧠 Neural Networks Implemented

- **AlexNet**
- **VGGNet** 
- **ResNet** 

These models were implemented using **PyTorch** and **TensorFlow** frameworks.

---

## 🔧 Methodology

1. **Data Preprocessing**
   - Normalization
   - One-hot encoding of labels
   - Data augmentation for CIFAR-10

2. **Model Training**
   - Transfer learning or training from scratch
   - Optimizers used: Adam, SGD
   - Loss function: CrossEntropyLoss

3. **Evaluation**
   - Accuracy and loss tracking
   - Confusion matrix
   - Training/validation plots

---

## 📊 Results

| Model     | MNIST Accuracy | CIFAR-10 Accuracy |
|-----------|----------------|-------------------|
| AlexNet   | ~98%           | ~85%              |
| VGGNet    | ~98.5%         | ~88%              |
| ResNet    | ~99%           | ~90%              |

> ⚠️ *Actual results may vary depending on training settings and compute resources.*

---

## 📁 Requirements

- `numpy`
- `matplotlib`
- `torch` / `tensorflow`
- `torchvision`
- `scikit-learn`

Install using pip:

```bash
pip install numpy matplotlib torch torchvision scikit-learn tensorflow
```

---



## 📄 Output

- Accuracy and loss plots
- Confusion matrices
- Trained model weights

---


# 🚀 CIFAR-10 Image Classification using GoogLeNet

A deep learning project that fine-tunes a pre-trained **GoogLeNet
(Inception v1)** model using **PyTorch** to classify images from the
**CIFAR-10** dataset into 10 object categories.

## 📋 Overview

This project uses **transfer learning** on GoogLeNet to classify images
into:

✈️ Airplane, 🚗 Automobile, 🐦 Bird, 🐱 Cat, 🦌 Deer, 🐕 Dog, 🐸 Frog,
🐴 Horse, 🚢 Ship, 🚚 Truck

## ✨ Features

-   ✅ Pre-trained GoogLeNet (ImageNet)
-   🎨 Data Augmentation
-   ⚡ GPU Support (CUDA auto-detect)
-   🔁 Transfer Learning
-   🧠 Adam Optimizer

### Clone & Install

``` bash
git clone <your-repo-url>
cd cifar10-googlenet-classifier
pip install -r requirements.txt
```

## 📦 Dataset

Auto-download CIFAR-10: - 50k train images, 10k test images - 32×32
resized to 224×224

## 🏃‍♂️ Run Training

``` bash
python cnn_classifier.py
```

### Modify Hyperparameters

``` python
batch_size = 64
epochs = 5
learning_rate = 0.0005
```

## 🧠 Architecture

    GoogLeNet
    ├── Inception Modules
    ├── Global Avg Pool
    └── FC (1024 → 10)

## ⚙️ Training Settings

  Setting      Value
  ------------ ------------------
  Loss         CrossEntropyLoss
  Optimizer    Adam
  LR           0.0005
  Batch Size   64
  Epochs       5

## 📊 Expected Results

| Accuracy \| 90--95% \|
| Runtime \| \~10--15 mins/epoch (GPU) \|

## 📂 Structure

    ├── cnn_classifier.py
    ├── requirements.txt
    ├── README.md
    └── data/

## 🔧 Troubleshooting

-   Reduce `batch_size` if memory error
-   Check GPU:

``` python
print(next(model.parameters()).device)
```

## 🔮 Future Work

-   Save best model
-   Confusion matrix
-   Early stopping
-   ResNet/VGG support

## 📚 References

-   PyTorch Docs\
-   GoogLeNet Paper\
-   CIFAR-10 Dataset

## 📄 License

MIT License


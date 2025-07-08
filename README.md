# 🧠 EfficientNetB0 Transfer Learning on CIFAR-10

A deep learning project using **EfficientNetB0** pretrained on ImageNet to classify CIFAR-10 images with high accuracy via transfer learning and fine-tuning.

---

## 📊 Dataset

- **CIFAR-10**: 60,000 32x32 color images across 10 classes.
- Resized to 224x224 to fit EfficientNetB0 input size.

---

## 🔧 Model Architecture

- Base: `EfficientNetB0` (`include_top=False`, pretrained on ImageNet)
- GlobalAveragePooling → Dense(128, ReLU, L2) → Dropout → Dense(10, Softmax, L1)

---

## 🔄 Training Flow

1. Load and resize CIFAR-10 data.
2. Freeze base model and train top layers.
3. Unfreeze last 20 layers of EfficientNetB0.
4. Fine-tune with a lower learning rate.

---

## 📈 Performance

- Trained on 80% of training data.
- Fine-tuned last 20 layers for improved accuracy.
- Evaluated on test set with ~10,000 images.

---

## 🚀 Run the Code

```bash
pip install tensorflow

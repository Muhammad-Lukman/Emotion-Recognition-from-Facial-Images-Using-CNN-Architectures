# Emotion Recognition from Facial Images using CNN Architectures

This project focuses on **classifying human emotions (Angry, Happy, Sad)** from facial images using **Convolutional Neural Networks (CNNs)**.  
The dataset was preprocessed, augmented, and trained using TensorFlow/Keras with performance improvements through **class balancing, callbacks, and checkpointing**.

---

## Repository Structure
```bash
Emotion-Recognition-from-Facial-Images-Using-CNN-Architectures/
│
├── notebooks/
│   └── Human_Emotions_Detection.ipynb     # code
│
├── models/
│   └── EmotionModelBest.keras             # Final trained model
│
├── results/
│   ├── training_accuracy_loss.png         # Accuracy & loss curves
│   └── confusion_matrix.png               # Confusion matrix plot
│
├── requirements.txt                       # Dependencies (TensorFlow, cv2, etc.)
├── README.md                              # Project overview, usage instructions
└── .gitignore                             # Ignore unnecessary files
```

---

## Learnings
- CNN model for facial emotion recognition (Angry, Happy, Sad).
- Handled **class imbalance** using weighted training.
- Integrated **EarlyStopping, Learning Rate Scheduler, and Checkpoints**.
- Final model achieves strong accuracy across classes.

---

## Dataset
- Dataset: 2,278 facial images (may increase after augmentation) across 3 classes:
  - Angry: 515  
  - Happy: 1006  
  - Sad: 757  
- Images resized and normalized for CNN input.

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("muhammadhananasghar/human-emotions-datasethes")

print("Path to dataset files:", path)
```

---

## Model Training
- Framework: **TensorFlow/Keras**
- Optimizer: `Adam`
- Loss: `Categorical Crossentropy`
- Key callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Class weights applied to mitigate dataset imbalance.

---

## Prediction Example

```python
import cv2
import tensorflow as tf

# Load image
img = cv2.imread("path_to_image.jpg")
img = tf.constant(img, dtype = tf.float32)
img = tf.expand_dims(img, axis=0) # Add batch dimension

# Predict
model = tf.keras.models.load_model("./models/EmotionModelBest.keras")
class_names = ['angry', 'happy', 'sad']
pred = class_names[tf.argmax(model.predict(img), axis=-1).numpy()[0]]
print("Predicted Emotion:", pred)
```

## Usage

1. Clone the repository:

```python
git clone https://github.com/your-username/Emotion-Recognition-from-Facial-Images-Using-CNN-Architectures.git
cd Emotion-Recognition-from-Facial-Images-Using-CNN-Architectures

```
2. Install dependencies:
```python
pip install -r requirements.txt

```
## Future Work
Above model is just a Basic model, though it has high accuracy & recall for validation data but it still confuses Happy with other classes. So, I'll refine it in near future and will upload that model also.
- Will try to Apply transfer learning (VGG16, ResNet, MobileNet).
- and also will try to Expand it to more emotions (neutral, surprise, fear, disgust).





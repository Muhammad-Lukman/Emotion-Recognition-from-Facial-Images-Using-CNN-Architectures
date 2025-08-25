# Emotion Recognition from Facial Images using CNN Architectures

This project focuses on **classifying human emotions (Angry, Happy, Sad)** from facial images using **Convolutional Neural Networks (CNNs)**.  
The dataset was preprocessed, augmented, and trained using TensorFlow/Keras with performance improvements through **class balancing, callbacks, and checkpointing**.

---

## Historical Context
The CNN used in this project is inspired by **LeNet-5**, one of the earliest successful convolutional neural networks, developed by **Yann LeCun et al. in 1998** for handwritten digit recognition (MNIST dataset). 

- **LeNet-5** was created by **Yann LeCun**, considered the “father of CNNs,” in 1998.  
- It was originally designed to classify handwritten digits (MNIST dataset).  
- LeNet’s success paved the way for modern CNNs like AlexNet, VGG, ResNet, and EfficientNet.  
- Our model follows a similar **conv–pool–conv–pool–fc–softmax** design but is adapted for **facial emotion recognition**.
  
---

## Model Architecture
 
Although we adapted LeNet for emotion recognition, the structure retains the same principles:

- **Convolutional Layers** – Learn spatial features (edges, textures, facial expressions).  
- **Pooling Layers** – Reduce spatial dimensions while keeping important features.  
- **Fully Connected Layers** – Perform classification based on extracted features.  
- **Softmax Output** – Classifies the face into one of the 3 emotions: *Angry, Happy, Sad*.
  
The model has **following modern modifications**:

- Batch Normalization for faster convergence and stable training.
- Dropout layers to reduce overfitting.
- L2 regularization for weight control.
- Flexible input resizing so the model accepts arbitrary input sizes ((None, None, 3)), rescaled to a fixed dimension. 

### Layer-wise breakdown of our model

1. **Input Layer**: Accepts RGB images of arbitrary size.
2. **Resizing & Rescaling**: Normalizes images to 256×256 with values scaled between 0–1.
3. **Conv Layer 1**: Conv2D(filters=6, kernel_size=3, strides=1, activation='relu')
4. **BatchNorm → MaxPooling → Dropout**
5. **Conv Layer 2**: Conv2D(filters=16, kernel_size=3, strides=1, activation='relu')
6. **BatchNorm → MaxPooling**
7. **Flatten**
8. **Dense Layer 1**: 100 units, ReLU + BatchNorm + Dropout
9. **Dense Layer 2**: 10 units, ReLU + BatchNorm + Dropout
10. **Output Layer**: 3 units, Softmax (for Angry, Happy, Sad)

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

---

## Learnings
- CNN model for facial emotion recognition (Angry, Happy, Sad).
- Handled **class imbalance** using weighted training.
- Integrated **EarlyStopping, Learning Rate Scheduler, and Checkpoints**.
- Final model achieves strong accuracy across classes.

## Future Work
Above model is just a Basic model, though it has high accuracy & recall for validation data but it still confuses Happy with other classes. So, I'll refine it in near future and will upload that model also.
- Will try to Apply transfer learning (VGG16, ResNet, MobileNet).
- and also will try to Expand it to more emotions (neutral, surprise, fear, disgust).





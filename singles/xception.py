import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import skimage.io
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.metrics import AUC
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, Flatten, Dropout,BatchNormalization ,Activation
from tensorflow.keras.models import Model, Sequential
from keras.applications.nasnet import NASNetLarge
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

import os

# Set your base directory path
base_dir = '/Users/apple/Desktop/hira/h-vgxe/dataset'

# Subdirectories
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Emotion classes (matching lowercase folder names)
categories = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Function to count images in each category
def count_images(directory):
    counts = {}
    for category in categories:
        category_path = os.path.join(directory, category)
        if os.path.exists(category_path):
            counts[category.capitalize()] = len([f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        else:
            counts[category.capitalize()] = 0
    return counts

# Print image counts
print("Training Set:", count_images(train_dir))
print("Test Set:", count_images(test_dir))

import pandas as pd

# Create a DataFrame for visualization
data = []
for dataset_name, dataset_dir in [('Train', train_dir), ('Test', test_dir)]:
    counts = count_images(dataset_dir)
    for emotion, count in counts.items():
        data.append({'Dataset': dataset_name, 'Emotion': emotion, 'Count': count})

df = pd.DataFrame(data)

# Plot using Seaborn with proper DataFrame structure
plt.figure(figsize=(12, 6))
sn.barplot(data=df, x='Emotion', y='Count', hue='Dataset')
plt.title('Emotion Distribution Across Datasets')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

IMG_SIZE = 200
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Use 20% of training data for validation
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    subset='training'
)

val_generator = val_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)


# Get a batch of augmented images
aug_images, aug_labels = next(train_generator)

# Plot
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(aug_images[i].squeeze(), cmap='gray')  # Remove channel dim for grayscale
    plt.title(list(train_generator.class_indices.keys())[np.argmax(aug_labels[i])])
    plt.axis('off')
plt.tight_layout()
plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import AdamW

IMG_SIZE = 200
NUM_CLASSES = 7

# Load Xception base model
base_model = Xception(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze layers
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Create the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    # Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
    # Dropout(0.5),
    # Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    # Dropout(0.6),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.4),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-4), # Example AdamW
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=2, factor=0.3, min_lr=1e-6, verbose=1)
checkpoint = ModelCheckpoint('best_xception_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)

# Summary
model.summary()

fine_tune_history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=16,
    callbacks=[early_stop, reduce_lr, checkpoint],
)


# Evaluate on the test set
test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"\n Test Accuracy: {test_acc:.4f}")
print(f" Test Loss: {test_loss:.4f}")

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Plot training and validation accuracy/loss
history = fine_tune_history.history

plt.figure(figsize=(14, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], 'b',label='Train Accuracy')
plt.plot(history['val_accuracy'], 'r--',label='Val Accuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history['loss'], 'b' , label='Train Loss')
plt.plot(history['val_loss'], 'r--',label='Val Loss')
plt.title('Model Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Predict on the test set
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

# Get class labels
class_labels = list(test_generator.class_indices.keys())

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Classification Report
report = classification_report(y_true, y_pred, target_names=class_labels)
print("Classification Report:\n")
print(report)
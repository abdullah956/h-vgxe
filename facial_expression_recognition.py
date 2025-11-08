"""
Facial Expression Recognition Model
Complete code extracted from 99.ipynb
"""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2
import matplotlib.image as mpimg
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10

import tensorflow as tf
from skimage.color import rgb2gray
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import (
    Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, 
    Conv2DTranspose, MaxPooling2D, concatenate, Flatten, Dense, Reshape, 
    Multiply, Add, UpSampling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import plot_model, np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow.keras as keras
import keras.layers as layers

# Optional: GradCAM for visualization (requires tf_explain)
try:
    from tf_explain.core.grad_cam import GradCAM
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("Warning: tf_explain not available. GradCAM visualization will be skipped.")

print("Tensorflow version:", tf.__version__)

# ============================================================================
# CUSTOM LAYER
# ============================================================================

class StandardizedConv2DWithOverride(layers.Conv2D):
    def convolution_op(self, inputs, kernel):
        mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)
        return tf.nn.conv2d(
            inputs,
            (kernel - mean) / tf.sqrt(var + 1e-10),
            padding="VALID",
            strides=list(self.strides),
            name=self.__class__.__name__,
        )

# ============================================================================
# DATA LOADING
# ============================================================================

data_path = 'dataset'
data_dir_list = os.listdir(data_path)

img_rows = 256
img_cols = 256
num_channel = 1
num_epoch = 10

img_data_list = []
labels_list = []  # Track labels as we load images

# Map emotion folder names to label indices
# Label order: ['surprise','fear','sadness','disgust','contempt','happy','anger']
emotion_to_label = {
    'surprise': 0,
    'fear': 1,
    'sad': 2,      # sadness
    'sadness': 2,
    'disgust': 3,
    'neutral': 4,  # mapped to contempt
    'contempt': 4,
    'happy': 5,
    'angry': 6,    # anger
    'anger': 6
}

# Handle nested directory structure: dataset/train/emotion/*.jpg and dataset/test/emotion/*.jpg
for dataset in data_dir_list:  # dataset is 'train' or 'test'
    emotion_dirs = os.listdir(data_path + '/' + dataset)
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    
    for emotion in emotion_dirs:  # emotion is 'angry', 'disgust', etc.
        emotion_path = os.path.join(data_path, dataset, emotion)
        
        # Skip if it's not a directory
        if not os.path.isdir(emotion_path):
            continue
        
        # Get label for this emotion
        emotion_lower = emotion.lower()
        if emotion_lower not in emotion_to_label:
            print(f'  Warning: Unknown emotion "{emotion}", skipping...')
            continue
        
        label = emotion_to_label[emotion_lower]
            
        img_list = os.listdir(emotion_path)
        print('  Processing emotion: {} ({} images) -> label {}'.format(emotion, len(img_list), label))
        
        for img in img_list:
            img_path = os.path.join(emotion_path, img)
            
            # Skip if it's a directory or not an image file
            if os.path.isdir(img_path):
                continue
            if not img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            
            input_img = cv2.imread(img_path)
            
            # Check if image was loaded successfully
            if input_img is None:
                print(f'  Warning: Could not read image {img_path}')
                continue
            
            #input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            input_img_resize = cv2.resize(input_img, (48, 48))
            img_data_list.append(input_img_resize)
            labels_list.append(label)  # Store label for this image
        
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data/255
print(f"Image data shape: {img_data.shape}")

# ============================================================================
# LABEL PROCESSING
# ============================================================================

num_classes = 7

# Use automatically generated labels from image loading
labels = np.array(labels_list, dtype='int64')
num_of_samples = img_data.shape[0]

print(f'Total samples: {num_of_samples}')
print(f'Label distribution:')
for i, name in enumerate(['surprise','fear','sadness','disgust','contempt','happy','anger']):
    count = np.sum(labels == i)
    print(f'  {name} (label {i}): {count} images')

names = ['surprise','fear','sadness','disgust','contempt','happy','anger']

def getLabel(id):
    return ['surprise','fear','sadness','disgust','contempt','happy','anger'][id]

# ============================================================================
# DATA PREPARATION
# ============================================================================

Y = np_utils.to_categorical(labels, num_classes)

# Shuffle the dataset
x, y = shuffle(img_data, Y, random_state=2)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=2)
x_test = X_test

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# ============================================================================
# MODEL 1: CUSTOM CNN
# ============================================================================

# Inputs
input_layer = Input((48, 48, 3))

# Encoder
f1 = StandardizedConv2DWithOverride(32, kernel_size=3, strides=1, padding='same', activation='relu')(input_layer)
f1 = BatchNormalization()(f1)
f = StandardizedConv2DWithOverride(32, kernel_size=3, strides=1, padding='same', activation='relu')(f1)
f = StandardizedConv2DWithOverride(32, kernel_size=3, strides=1, padding='same', activation='relu')(f)
f = MaxPooling2D(2, 2)(f)
f2 = Conv2D(32, kernel_size=1, strides=2, padding='same', activation='relu')(f)

f1 = StandardizedConv2DWithOverride(32, kernel_size=5, strides=1, padding='same', activation='relu')(f1)
f1 = StandardizedConv2DWithOverride(32, kernel_size=5, strides=2, padding='same', activation='relu')(f1)

f = concatenate([f, f1])
f = BatchNormalization()(f)

f1 = StandardizedConv2DWithOverride(64, kernel_size=3, strides=1, padding='same', activation='relu')(f)
f = StandardizedConv2DWithOverride(64, kernel_size=3, strides=1, padding='same', activation='relu')(f1)
f = StandardizedConv2DWithOverride(64, kernel_size=3, strides=1, padding='same', activation='relu', name='BeforeFinal_Layer')(f)
f = MaxPooling2D(2, 2)(f)
f3 = Conv2D(32, kernel_size=1, strides=2, padding='same', activation='relu')(f)

f1 = StandardizedConv2DWithOverride(64, kernel_size=5, strides=1, padding='same', activation='relu')(f1)
f1 = StandardizedConv2DWithOverride(64, kernel_size=5, strides=2, padding='same', activation='relu')(f1)

f = concatenate([f, f1])
f = BatchNormalization()(f)
                    
f1 = StandardizedConv2DWithOverride(128, kernel_size=3, strides=1, padding='same', activation='relu')(f)
f = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(f1)
f = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(f)
f = MaxPooling2D(2, 2)(f)
f4 = Conv2D(32, kernel_size=1, strides=2, padding='same', activation='relu')(f)

f1 = StandardizedConv2DWithOverride(128, kernel_size=5, strides=1, padding='same', activation='relu')(f1)
f1 = StandardizedConv2DWithOverride(128, kernel_size=5, strides=2, padding='same', activation='relu')(f1)
f1 = BatchNormalization()(f1)

f = concatenate([f, f1])

f1 = Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu')(f)
f = Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu')(f1)
f = Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu')(f)
f = MaxPooling2D(2, 2)(f)

f1 = Conv2D(512, kernel_size=3, strides=2, padding='same', activation='relu')(f1)
f1 = BatchNormalization()(f1)

f = concatenate([f, f1])
f = StandardizedConv2DWithOverride(512, kernel_size=3, strides=1, padding='same', activation='relu', name='Final_Layer')(f)
f = BatchNormalization()(f)
                    
f = Flatten()(f)
f = Dropout(rate=0.3)(f)
f = BatchNormalization()(f)
f = Dense(512, activation='relu')(f)
f = Dropout(rate=0.32)(f)
f = BatchNormalization()(f)
output_layer = Dense(7, activation='softmax')(f)

# Model
model = Model(
    inputs=[input_layer],
    outputs=[output_layer]
)
model.summary()

# Save model architecture diagram (optional)
try:
    plot_model(model, to_file='model.png', show_shapes=True)
    print("Model architecture saved to model.png")
except Exception as e:
    print(f"Could not save model diagram: {e}")

# ============================================================================
# OPTIONAL: ShowProgress Callback (for GradCAM visualization during training)
# ============================================================================

IMAGE_SIZE = (48, 48)
label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

class ShowProgress(keras.callbacks.Callback):
    """
    Optional callback to show GradCAM visualization during training.
    Requires a test image path to be specified.
    """
    def __init__(self, test_image_path=None):
        super().__init__()
        self.test_image_path = test_image_path
    
    def on_epoch_end(self, epoch, logs=None):
        if not GRADCAM_AVAILABLE or self.test_image_path is None:
            return
        
        try:
            from keras.preprocessing import image
            img = image.load_img(self.test_image_path, target_size=(48, 48), color_mode="grayscale")
            img = np.array(img)
            img1 = np.expand_dims(img, axis=0)  # makes image shape (1,48,48)
            img2 = img1.reshape(1, 48, 48, 3)
            result = self.model.predict(img2)
            result = list(result[0])
            img_index = result.index(max(result))
            label = label_dict[img_index]
            
            exp = GradCAM()
            cam = exp.explain(
                validation_data=(img2, result),
                class_index=1,
                layer_name='Final_Layer',
                model=self.model
            )
            
            plt.figure(figsize=(5, 2))
            plt.subplot(1, 2, 1)
            plt.imshow(np.squeeze(img))
            plt.title('Original Image')
            
            plt.subplot(1, 2, 2)
            plt.title('GradCAM')
            plt.imshow(cam)
            plt.tight_layout()
            plt.show()
            print("***************************Predicted label: ", label)
        except Exception as e:
            print(f"Error in ShowProgress callback: {e}")

# ============================================================================
# MODEL TRAINING
# ============================================================================

input_shape = (48, 48, 3)  # Input shape definition
epochs = 300
learning_rate = 1e-3
opt = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    ModelCheckpoint('model.hdf5', monitor="val_accuracy", verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=20, verbose=1, min_lr=1e-6),
    # ShowProgress(),  # Uncomment and provide test_image_path to enable GradCAM visualization
    EarlyStopping(monitor='val_accuracy', restore_best_weights=True, patience=100)
]

print("\nStarting training...")
history = model.fit(
    X_train, y_train, 
    batch_size=7, 
    epochs=epochs, 
    verbose=1, 
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

# ============================================================================
# VISUALIZE TRAINING HISTORY
# ============================================================================

train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs_range = range(len(train_acc))

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss, 'r', label='train_loss')
plt.plot(epochs_range, val_loss, 'b', label='val_loss')
plt.title('Train Loss vs Val Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_acc, 'r', label='train_acc')
plt.plot(epochs_range, val_acc, 'b', label='val_acc')
plt.title('Train Acc vs Val Acc')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# ============================================================================
# MODEL EVALUATION
# ============================================================================

score = model.evaluate(X_test, y_test, verbose=0)
print('\nModel Evaluation:')
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

# Test on a single image
test_image = X_test[0:1]
print(f'\nTest image shape: {test_image.shape}')
print('Predictions:', model.predict(test_image))

classes_x = np.argmax(model.predict(test_image), axis=1)
print('True label:', y_test[0:1])

# Visualize predictions on 9 test images
res = np.argmax(model.predict(X_test[0:9]), axis=1)
plt.figure(figsize=(10, 10))

for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_test[i], cmap=plt.get_cmap('gray'))
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    plt.ylabel('prediction = %s' % getLabel(res[i]), fontsize=14)

plt.tight_layout()
plt.savefig('test_predictions.png')
plt.show()

# ============================================================================
# CONFUSION MATRIX
# ============================================================================

results = model.predict(X_test)
predicted_classes = np.argmax(results, axis=1)
y_true = np.argmax(y_test, axis=1)

# Define emotion labels
label = ['sad', 'happy', 'disgust', 'angry', 'neutral', 'fear', 'surprise']
labels_dict = {0: 'sad', 1: 'happy', 2: 'disgust', 3: 'angry', 4: 'neutral', 5: 'fear', 6: 'surprise'}

# Calculate confusion matrix
cm = confusion_matrix(y_true, predicted_classes)

# Transform to DataFrame for easier plotting
cm_df = pd.DataFrame(cm, index=label, columns=label)

# Create heatmap
plt.figure(figsize=(7, 7))
sns.heatmap(cm_df, annot=True, cmap='Blues', cbar=False, linewidth=2, fmt='d')
plt.title('Emotion Classification Confusion Matrix')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# ============================================================================
# CLASSIFICATION REPORT
# ============================================================================

target_names = ['sad', 'happy', 'disgust', 'angry', 'neutral', 'fear', 'surprise']
report = classification_report(y_true, predicted_classes, target_names=target_names)
print('\nClassification Report:')
print(report)

# ============================================================================
# MODEL 2: ALTERNATE CNN ARCHITECTURE (Optional - different from Model 1)
# ============================================================================

# This is an alternate model architecture from Cell 12
# Uncomment to use this instead of Model 1

"""
# Alternate Model Architecture
input_layer_alt = Input((48, 48, 3))

# Encoder
f1 = StandardizedConv2DWithOverride(32, kernel_size=3, strides=3, padding='same', activation='relu')(input_layer_alt)
f1 = BatchNormalization()(f1)
f = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(f1)
f = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(f)
f = MaxPooling2D(2, 2)(f)
f2 = Conv2D(32, kernel_size=1, strides=2, padding='same', activation='relu')(f)

f1 = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(f1)
f1 = Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')(f1)

f = concatenate([f, f1])
f = BatchNormalization()(f)

f1 = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(f)
f = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(f1)
f = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu', name='BeforeFinal_Layer')(f)
f = MaxPooling2D(2, 2)(f)
f3 = Conv2D(32, kernel_size=1, strides=2, padding='same', activation='relu')(f)

f1 = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(f1)
f1 = Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu')(f1)

f = concatenate([f, f1, f2])
f = BatchNormalization()(f)
                    
f1 = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(f)
f = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(f1)
f = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(f)
f = MaxPooling2D(2, 2)(f)
f4 = Conv2D(32, kernel_size=1, strides=2, padding='same', activation='relu')(f)

f1 = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(f1)
f1 = Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu')(f1)
f1 = BatchNormalization()(f1)

f = concatenate([f, f1, f3])

f1 = Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu')(f)
f = Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu')(f1)
f = Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu')(f)
f = MaxPooling2D(2, 2)(f)

f1 = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(f1)
f1 = Conv2D(256, kernel_size=3, strides=2, padding='same', activation='relu')(f1)
f1 = BatchNormalization()(f1)

f = concatenate([f, f1, f4])
f = Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu', name='Final_Layer')(f)
f = BatchNormalization()(f)
                    
f = Flatten()(f)
f = Dropout(rate=0.3)(f)
f = Dense(1024, activation='relu')(f)
f = Dropout(rate=0.32)(f)
output_layer_alt = Dense(7, activation='softmax')(f)

# Alternate Model
model_alt = Model(
    inputs=[input_layer_alt],
    outputs=[output_layer_alt]
)
model_alt.summary()
"""

# ============================================================================
# MODEL 3: RESNET50
# ============================================================================

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D

print("\n" + "="*50)
print("Building ResNet50 Model")
print("="*50)

# Load the pre-trained ResNet50 model without the top (fully connected) layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Create a new sequential model for your top layers
model_resnet = keras.Sequential()

# Add the ResNet50 base model
model_resnet.add(base_model)

# Add a Global Average Pooling layer to reduce the spatial dimensions
model_resnet.add(GlobalAveragePooling2D())

# Add a fully connected layer with ReLU activation
model_resnet.add(Dense(512, activation='relu'))

# Add dropout for regularization
model_resnet.add(Dropout(0.5))

# Output layer with 7 classes for facial expressions
model_resnet.add(Dense(7, activation='softmax'))

# Compile the ResNet-based model
model_resnet.compile(
    optimizer=Adam(lr=0.0001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

model_resnet.summary()

# Train the ResNet-based model
print("\nTraining ResNet50 model...")
history_resnet = model_resnet.fit(
    X_train, y_train, 
    epochs=100, 
    validation_data=(X_test, y_test), 
    callbacks=callbacks
)

# Evaluate the ResNet-based model
resnet_scores = model_resnet.evaluate(X_test, y_test, verbose=0)
resnet_accuracy = resnet_scores[1]
resnet_loss = resnet_scores[0]

print("\nResNet Model Evaluation:")
print("ResNet Model - Accuracy: {:.2f}% | Loss: {:.4f}".format(resnet_accuracy * 100, resnet_loss))

# ============================================================================
# TEST PREDICTION WITH RESNET
# ============================================================================

test_image_index = 13
test_image = X_test[test_image_index]
true_label = getLabel(np.argmax(y_test[test_image_index]))

# Make predictions on the test image using the ResNet-based model
predictions = model_resnet.predict(np.expand_dims(test_image, axis=0))
predicted_label_index = np.argmax(predictions)
predicted_label = getLabel(predicted_label_index)

# Display the test image and its predicted label
plt.figure(figsize=(5, 5))
plt.imshow(test_image.squeeze(), cmap='gray')
plt.title(f"True Label: {true_label}\nPredicted Label: {predicted_label}")
plt.axis('off')
plt.tight_layout()
plt.savefig('resnet_prediction.png')
plt.show()

print(f"\nTrue Label: {true_label}")
print(f"Predicted Label: {predicted_label}")

print("\n" + "="*50)
print("Script completed successfully!")
print("="*50)


import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow_model_optimization.sparsity import keras as sparsity
import matplotlib.pyplot as plt
import shutil

# ------------------ 1. DATASET PREPARATION ------------------

# Define dataset paths
DATASETS = {
    "NIH": "path_to_NIH_dataset",
    "COVID": "path_to_COVID_dataset",
    "Pneumonia": "path_to_Pneumonia_dataset"
}

# Create a unified dataset folder
os.makedirs("dataset/train", exist_ok=True)
os.makedirs("dataset/val", exist_ok=True)
os.makedirs("dataset/test", exist_ok=True)

# Preprocess and save images
def preprocess_and_save(source_folder, target_folder, img_size=(224, 224)):
    for file in os.listdir(source_folder):
        img = cv2.imread(os.path.join(source_folder, file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size)
        img = cv2.equalizeHist(img)  # Enhance contrast
        cv2.imwrite(os.path.join(target_folder, file), img)

for dataset, path in DATASETS.items():
    preprocess_and_save(path, "dataset/train")

# Train-validation-test split
from sklearn.model_selection import train_test_split
all_files = os.listdir("dataset/train")
train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=42)

for f in val_files:
    shutil.move(f"dataset/train/{f}", "dataset/val/")
for f in test_files:
    shutil.move(f"dataset/train/{f}", "dataset/test/")

# ------------------ 2. DATA AUGMENTATION ------------------
datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, 
    horizontal_flip=True, rescale=1./255
)

# ------------------ 3. BUILD LIGHTWEIGHT MODEL ------------------
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:100]:  
    layer.trainable = False  # Freeze lower layers for efficiency

# Image Input
image_input = Input(shape=(224, 224, 3))
x = base_model(image_input, training=False)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

# Metadata Input (Age, Symptoms)
metadata_input = Input(shape=(3,))
y = Dense(8, activation='relu')(metadata_input)

# Merge Inputs
merged = Concatenate()([x, y])
output = Dense(1, activation='sigmoid')(merged)

# Define Model
model = Model(inputs=[image_input, metadata_input], outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ------------------ 4. TRAINING WITH PRUNING ------------------
pruning_params = {
    "pruning_schedule": sparsity.PolynomialDecay(initial_sparsity=0.2, final_sparsity=0.8, begin_step=2000, end_step=4000)
}

pruned_model = sparsity.prune_low_magnitude(model, **pruning_params)

# Training
train_gen = datagen.flow_from_directory("dataset/train", target_size=(224,224), batch_size=32, class_mode="binary")
val_gen = datagen.flow_from_directory("dataset/val", target_size=(224,224), batch_size=32, class_mode="binary")

pruned_model.fit(train_gen, validation_data=val_gen, epochs=10)

# Save model
pruned_model = sparsity.strip_pruning(pruned_model)  # Remove pruning wrappers
pruned_model.save("optimized_pneumonia_model.h5")

# ------------------ 5. GRAD-CAM (EXPLAINABILITY) ------------------
def grad_cam(image_path, model):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=[0, -1])

    last_conv_layer = model.get_layer("top_conv")
    grad_model = tf.keras.models.Model([model.input], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_output)[0]
    weights = np.mean(grads, axis=(0, 1))
    heatmap = np.dot(conv_output[0], weights)

    plt.imshow(heatmap, cmap='jet')
    plt.colorbar()
    plt.show()

# ------------------ 6. CONVERT MODEL TO TENSORFLOW LITE (DEPLOYMENT) ------------------
converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
lite_model = converter.convert()

with open("pneumonia_model.tflite", "wb") as f:
    f.write(lite_model)

print("🚀 Model optimized and ready for deployment!")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder

# ✅ 1. Define dataset path (Make sure the path is correct for Windows)
dataset_path = (r"C:\Users\JAGGU'S PC\OneDrive\Desktop\html-css-course\skin_dataset")

# ✅ 2. Image size and batch size
batch_size = 32
img_size = (224, 224)  # Standard input size for CNN

# ✅ 3. Define ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values (0 to 1)
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 80% Training, 20% Validation
)

# ✅ 4. Load dataset
train_generator = datagen.flow_from_directory(
    dataset_path, 
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_path, 
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# ✅ 5. Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),  # First Conv Layer
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),  # Second Conv Layer
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Regularization
    Dense(train_generator.num_classes, activation='softmax')  # Output Layer
])

# ✅ 6. Compile Model
model.compile(optimizer=Adam(learning_rate=0.001),  
              loss='categorical_crossentropy',  
              metrics=['accuracy'])

# ✅ 7. Print Model Summary
model.summary()

# ✅ 8. Train the Model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size
)

# ✅ 9. Save the Model
model.save('skin_disease_model.h5')
print("Model Saved Successfully!")

# ✅ 10. Functions for Image Preprocessing & Prediction

def bilateral_filter(image):
    """ Apply Bilateral Filtering to Reduce Noise """
    return cv2.bilateralFilter(image, 9, 75, 75)

def enhance_contrast(image):
    """ Enhance Image Contrast Using CLAHE """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def adaptive_thresholding(image):
    """ Convert Image to Grayscale & Apply Adaptive Thresholding """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

def canny_edge_detection(image):
    """ Apply Canny Edge Detection """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 100, 200)

def preprocess_image(image_path):
    """ Load and Preprocess Image for Prediction """
    image = cv2.imread(image_path)
    image = bilateral_filter(image)
    image = enhance_contrast(image)
    image = cv2.resize(image, (224, 224))  # Resize to match model input
    image = image / 255.0  # Normalize
    return np.expand_dims(image, axis=0)  # Expand dims for model input

def detect_skin_disease(image_path, model_path='skin_disease_model.h5'):
    """ Predict Skin Disease from an Image """
    
    # Load the image
    image = cv2.imread(image_path)
    
    # ✅ Check if the image was loaded successfully
    if image is None:
        raise FileNotFoundError(f"Error: Could not load image at path '{image_path}'. Check if the file exists and the path is correct.")

    # Apply bilateral filter to reduce noise
    image_filtered = bilateral_filter(image)
    
    # Enhance the contrast of the image
    image_contrast_enhanced = enhance_contrast(image_filtered)
    
    # Segment the image using adaptive thresholding
    image_segmented = adaptive_thresholding(image_contrast_enhanced)
    
    # Apply Canny edge detection
    image_edges = canny_edge_detection(image_filtered)
    
    # Load the pre-trained CNN model
    model = load_model(model_path)
    
    # Extract features using the CNN model
    predictions = feature_extraction_cnn(image_filtered, model)
    
    # Decode the predictions
    label_encoder = LabelEncoder()
    predicted_class = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
    
    return predicted_class, image_edges

# ✅ 11. Example Prediction
image_path = r"C:\Users\JAGGU'S PC\OneDrive\Desktop\html-css-course\test_image.jpg"  # Replace with actual image path
predicted_disease = detect_skin_disease(image_path)
print(f"Predicted Skin Disease: {predicted_disease}")
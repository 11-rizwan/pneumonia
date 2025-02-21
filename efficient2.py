import tensorflow as tf from tensorflow import keras from tensorflow.keras import layers from tensorflow.keras.applications import EfficientNetB0 from tensorflow.keras.preprocessing.image import ImageDataGenerator import matplotlib.pyplot as plt

Set Parameters

IMG_SIZE = (224, 224) BATCH_SIZE = 32 EPOCHS = 20 DATASET_PATH = "dataset_path_here"

Data Augmentation & Preprocessing

data_gen = ImageDataGenerator( rescale=1./255, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, validation_split=0.2 )

train_gen = data_gen.flow_from_directory( DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', subset='training' )

val_gen = data_gen.flow_from_directory( DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', subset='validation' )

Load Pretrained EfficientNet-B0

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) base_model.trainable = False  # Freeze base model layers

Create Model

model = keras.Sequential([ base_model, layers.GlobalAveragePooling2D(), layers.Dense(256, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.4), layers.Dense(1, activation='sigmoid') ])

Compile Model

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

Train Model

history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

Evaluate Model

plt.plot(history.history['accuracy'], label='Train Accuracy') plt.plot(history.history['val_accuracy'], label='Validation Accuracy') plt.legend() plt.title("Model Accuracy") plt.show()

Save Model

model.save("pneumonia_detection_model.h5")

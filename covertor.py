import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("pneumonia_model.h5")

# Convert the model to TF-Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TF-Lite model
with open("pneumonia_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TF-Lite model saved as pneumonia_model.tflite")

converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# Save the quantized TF-Lite model
with open("pneumonia_model_quant.tflite", "wb") as f:
    f.write(tflite_quant_model)

print("Quantized TF-Lite model saved as pneumonia_model_quant.tflite")

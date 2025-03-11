import numpy as np
import tensorflow.lite as tflite
from tensorflow.keras.preprocessing import image

# Load the TF-Lite model
interpreter = tflite.Interpreter(model_path="pneumonia_model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess a sample image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

# Load and predict
img_array = preprocess_image("sample_xray.jpg")
interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()
prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

# Print result
result = "Pneumonia Detected" if prediction > 0.5 else "Normal"
print(f"Prediction: {result} (Confidence: {prediction:.4f})")

from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow.lite as tflite
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load TF-Lite model
interpreter = tflite.Interpreter(model_path="pneumonia_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess an image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Resize
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)  # Expand dims & convert
    return img_array

# Home route (renders HTML page)
@app.route('/')
def home():
    return render_template('index.html')

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    file_path = os.path.join("static/uploads", file.filename)
    file.save(file_path)

    # Preprocess the image
    img_array = preprocess_image(file_path)

    # Run inference with TF-Lite model
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # Interpret results
    result = "Pneumonia Detected" if prediction > 0.5 else "Normal"

    return render_template('index.html', prediction=result, confidence=f"{prediction:.4f}", image_url=file_path)

if __name__ == '__main__':
    app.run(debug=True)

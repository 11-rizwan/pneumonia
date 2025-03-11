from flask import Flask, request, jsonify
import numpy as np
import tensorflow.lite as tflite
from tensorflow.keras.preprocessing import image

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

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    file_path = "temp.jpg"
    file.save(file_path)

    # Preprocess the image
    img_array = preprocess_image(file_path)

    # Run inference with TF-Lite model
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # Interpret results
    result = "Pneumonia Detected" if prediction > 0.5 else "Normal"

    return jsonify({"prediction": result, "confidence": float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)





curl -X POST -F "file=@sample_xray.jpg" http://127.0.0.1:5000/predict

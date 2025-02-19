from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image

tf.config.set_visible_devices([], 'GPU')

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('pneumonia_model.h5')

# Prediction function
def predict_pneumonia(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (150, 150))  # Resize image
    img = img / 255.0  # Normalize image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    prediction = model.predict(img)
    return "Pneumonia Detected" if prediction[0][0] > 0.5 else "Normal"

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = "static/uploads/" + file.filename
            file.save(filepath)

            # Predict pneumonia from the uploaded image
            result = predict_pneumonia(filepath)
            return render_template("index.html", prediction=result, image_path=filepath)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)

import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your model
MODEL_PATH = 'hair_color_model.h5'
model = load_model(MODEL_PATH)

# Customize these as per your model's requirement
IMG_SIZE = (224, 224)  # Change if your model needs other input size
CLASS_NAMES = [
    "Black Hair",
    "Blonde Hair",
    "Gray/White Hair",
    "Red Hair"
]

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB').resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    img_url = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess and predict
            img = preprocess_image(filepath)
            preds = model.predict(img)[0]

            print("Preds:", preds)
            print("Preds shape:", np.shape(preds))
            print("Class names:", CLASS_NAMES)

            pred_class = "Unknown"
            confidence = 0

            if len(preds) == len(CLASS_NAMES):
                idx = int(np.argmax(preds))
                pred_class = CLASS_NAMES[idx]
                confidence = float(preds[idx]) * 100
            elif len(preds) == 1 and len(CLASS_NAMES) == 2:
                # Binary classification
                idx = int(preds[0] > 0.5)
                pred_class = CLASS_NAMES[idx]
                confidence = float(preds[0] if idx == 1 else 1 - preds[0]) * 100
            else:
                pred_class = f"Format output model/CLASS_NAMES tidak cocok ({len(preds)} prediksi, {len(CLASS_NAMES)} kelas)"

            result = f'{pred_class} ({confidence:.2f}%)'
            img_url = url_for('static', filename=f'uploads/{filename}')

    return render_template('index.html', result=result, img_url=img_url)

if __name__ == '__main__':
    app.run(debug=True)

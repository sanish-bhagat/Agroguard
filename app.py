from flask import Flask, request, jsonify, send_from_directory
import os
import numpy as np
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from dotenv import load_dotenv
import google.generativeai as genai  # ✅ Gemini API
import markdown as md

# --- Configuration & Model Loading ---
app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

IMAGE_SIZE = (256, 256)

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ Gemini API key not found in .env file")
genai.configure(api_key=GEMINI_API_KEY)

CLASS_NAMES = {
    "tomatoes": ["Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_healthy", "..."],
    "potatoes": ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy", "..."],
    "apple": ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy", "..."],
    "corn": ["Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", "..."],
    "peach": ["Peach___Bacterial_spot", "Peach___healthy", "..."]
}

# loading the model
MODELS = {}
for crop_name in CLASS_NAMES.keys():
    model_path = os.path.join('models', f'{crop_name}.h5')
    if os.path.exists(model_path):
        print(f"Loading model for {crop_name} from {model_path}...")
        MODELS[crop_name] = tf.keras.models.load_model(model_path, compile=False)
        print(f"Model for {crop_name} loaded.")
    else:
        print(f"Warning: Model file not found for {crop_name} at {model_path}")

print("All models loaded successfully!")

# image preprocessing
def preprocess_image(image_file):
    img = Image.open(image_file.stream).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route("/")
def serve_index():
    """Serve the frontend index.html"""
    return send_from_directory(app.static_folder, "index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    crop = request.form.get('crop')

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not crop or crop not in MODELS:
        return jsonify({'error': 'Invalid or unsupported crop type'}), 400

    try:
        processed_image = preprocess_image(file)
        model = MODELS[crop]
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0])) * 100
        predicted_class_name = CLASS_NAMES[crop][predicted_class_index]

        # --- LLM: Get treatment info from Gemini ---
        prompt = f"""
        You are an agricultural expert. Write a farmer-friendly plant disease guide in this exact Markdown + emoji style:

        # {predicted_class_name.replace('_', ' ')} in {crop} 🍎

        ## Symptoms 🔎
        * Use short bullet points
        * Keep language simple

        ## Causes 🦠
        * Explain clearly
        * Avoid technical jargon

        ## Organic Treatment 🌿
        * Use bold for product names
        * Short, actionable steps

        ## Chemical Treatment 🧪
        * Include timing
        * Mention safety precautions

        ## Prevention Tips 🍎
        * Actionable and clear

        **Disclaimer:** Always follow label instructions...
        """

        gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")
        llm_response = gemini_model.generate_content(prompt)

        treatment_info = llm_response.text
        html_treatment = md.markdown(treatment_info)

        return jsonify({
            'disease': predicted_class_name.replace('_', ' '),
            'confidence': f"{confidence:.2f}%",
            'treatment': html_treatment
        })
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

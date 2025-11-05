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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GEMINI_API_KEY:
#     raise RuntimeError("❌ Gemini API key not found in .env file")
# genai.configure(api_key=GEMINI_API_KEY)

if not GOOGLE_API_KEY:
    raise RuntimeError("❌ Gemini API key not found in .env file")
genai.configure(api_key=GOOGLE_API_KEY)

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
    """Serve the frontend predict.html"""
    return send_from_directory(app.static_folder, "predict.html")


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

        # --- Mock treatment info (API key invalid, using static response) ---
        treatment_info = f"""
        # {predicted_class_name.replace('_', ' ')} in {crop} 🍎

        ## Symptoms 🔎
        * Yellowing leaves
        * Spots on fruits

        ## Causes 🦠
        * Fungal infection

        ## Organic Treatment 🌿
        * Use **neem oil**

        ## Chemical Treatment 🧪
        * Apply fungicide as per instructions

        ## Prevention Tips 🍎
        * Proper watering

        **Disclaimer:** Consult a professional.
        """
        html_treatment = md.markdown(treatment_info)

        return jsonify({
            'disease': predicted_class_name.replace('_', ' '),
            'confidence': f"{confidence:.2f}%",
            'treatment': html_treatment
        })
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        context = data.get('context', '')

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Create a conversational AI model using Gemini
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Create a context-aware prompt
        prompt = f"""
        You are an agricultural expert chatbot helping farmers with plant disease diagnosis and treatment.
        The user has uploaded an image for disease analysis and is asking follow-up questions.

        Context: {context}
        User question: {user_message}

        Please provide helpful, accurate information about plant diseases, treatments, and agricultural practices.
        Keep your response concise but informative. Use simple language that farmers can understand.
        If the question is about treatment, emphasize both organic and chemical options when appropriate.
        Always remind users to consult professionals for severe cases.
        """

        response = model.generate_content(prompt)
        ai_response = response.text.strip()

        return jsonify({'response': ai_response})

    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({'error': 'Failed to process chat message'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

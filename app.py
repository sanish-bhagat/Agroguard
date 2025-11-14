from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import numpy as np
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from dotenv import load_dotenv
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
import config.tf_cpu


# --- Configuration & Model Loading ---
app = Flask(__name__, static_folder="templates", static_url_path="")
CORS(app)

IMAGE_SIZE = (256, 256)

# Load environment variables
load_dotenv()


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


PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

embeddings = download_hugging_face_embeddings()

index_name = "agroguard" 

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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

        # --- Query RAG for treatment recommendations ---
        treatment_query = f"What are the treatment recommendations for {predicted_class_name.replace('_', ' ')} in {crop}?"

        treatment_response = rag_chain.invoke({"input": treatment_query})
        memory.save_context({"input": treatment_query}, {"output": treatment_response["answer"]})

        treatment = treatment_response["answer"].strip()

        return jsonify({
            'disease': predicted_class_name.replace('_', ' '),
            'confidence': f"{confidence:.2f}%",
            'treatment': treatment
        })
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500
    

@app.route("/rag_chat_memory", methods=["POST"])
def rag_chat_memory():
    data = request.get_json()
    user_input = data.get("message", "")

    context = memory.load_memory_variables({})
    chat_history = context.get("chat_history", "")

    final_input = f"""
    Previous conversation:
    {chat_history}

    User question: {user_input}
    """

    response = rag_chain.invoke({"input": final_input})
    memory.save_context({"input": user_input}, {"output": response["answer"]})

    return jsonify({"response": response["answer"]})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
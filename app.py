from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

from dotenv import load_dotenv, dotenv_values, find_dotenv
from src.text_from_image_extractor import TextFromImageExtractor
from src.rag import RAG

# Load env variables to be used
load_dotenv(find_dotenv())
env_values = dotenv_values()
raw_data_folder = env_values['RAW_DATA_FOLDER']
processed_data_folder = env_values["PROCESSED_DATA_FOLDER"]
pdf_file_name = env_values["PDF_FILE_NAME"]
huggingface_token = env_values["HUGGINGFACE_TOKEN"]
milvus_port = env_values["MILVUS_DOCKER_PORT"]

# Flask app initial config
app = Flask(__name__)

app.config['RAW_DATA_FOLDER'] = raw_data_folder
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 # File limit to 5MB

# Create instances of TextFromImageExtractor and RAG to perform text extraction and question answering tasks.
text_from_image_extractor = TextFromImageExtractor(raw_data_folder=raw_data_folder, processed_data_folder=processed_data_folder)
rag = RAG(huggingface_token=huggingface_token, milvus_port=milvus_port)
file_path = raw_data_folder + pdf_file_name
rag.setup(file_path)


# Endpoint to accept PNG and name pairs
@app.route('/upload', methods=['POST'])
def upload_file():
    # Verify if 'file' is in request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400    
    file = request.files['file']

    # Verify the file format
    if all(file.filename.endswith(allowed_format) for allowed_format in [".jpeg", ".jpg", ".png"]):
        return jsonify({"error": "Invalid file format. Should be .jpeg, .jpg or .png"}), 400

    # Save file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['RAW_DATA_FOLDER'], filename)
    file.save(file_path)

    # Get JSON data from form
    first_name_real = request.form.get("first_name").title()
    last_name_real = request.form.get("last_name").title()

    # Get first and last names
    try:
        full_name_real = first_name_real + " " + last_name_real
    except:
        return jsonify({"error": "name_pairs should have 'first_name' and 'last_name' as keys"})

    # Extract text from submitted file
    result = text_from_image_extractor.text_extraction_and_landmarking(filename)
    
    # Perform fuzzy matching to calculate similarity score
    similarity_score = text_from_image_extractor.get_similarity_score(result["full_name"], full_name_real)
    
    # Build response in JSON format
    response = jsonify({
        "first_name": result["first_name"],
        "last_name": result["last_name"],
        "bonding_box_cords": result["box"],
        "similarity_score": similarity_score
    })
    return response, 200

# Endpoint to answer user's question based on given document
@app.route('/ask', methods=['POST'])
def respond_query():
    # Get question from form data
    question = request.form.get('question')

    # Run RAG to generate an answer
    answer = rag.run_rag(question)

    # Build response in JSON format
    response = jsonify({
        "question": question,
        "response": answer
    })
    return response, 200
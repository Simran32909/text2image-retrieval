from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import clip
import torch
app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

ALIGNED_EMBEDDINGS_PATH = r"data/aligned_embeddings.npz"
CAPTIONS_PATH = r"data/captions_tokenized.csv"
IMAGE_PATH_TEMPLATE = "static/images/flickr30k_images/{filename}"

aligned_data = np.load(ALIGNED_EMBEDDINGS_PATH)
text_emb = aligned_data["text"]
img_emb = aligned_data["image"]
metadata = pd.read_csv(CAPTIONS_PATH)
image_filenames = metadata["image"].tolist()
captions = metadata["clean_caption"].tolist()

def suggest_caption(user_input, captions, top_k=5):
    suggestions = [caption for caption in captions if user_input.lower() in caption.lower()]
    return suggestions[:top_k]

def text_to_image(query_text_emb, top_k=5):
    query_text_emb_cpu = query_text_emb.cpu().detach().numpy()
    similarities = cosine_similarity(query_text_emb_cpu.reshape(1, -1), img_emb).flatten()
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    results = [(image_filenames[i], similarities[i]) for i in top_k_indices]
    return results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/suggest_captions', methods=['POST'])
def suggest_captions_route():
    data = request.json
    user_input = data.get('text', '')
    if not user_input.strip():
        return jsonify({'error': 'Input text is required'}), 400
    suggestions = suggest_caption(user_input, captions)
    return jsonify({'suggestions': suggestions})

@app.route('/retrieve_images', methods=['POST'])
def retrieve_images_route():
    data = request.json
    text_input = data.get('text', '')
    if not text_input.strip():
        return jsonify({'error': 'Input text is required'}), 400
    text_embedding = model.encode_text(clip.tokenize([text_input]).to(device))
    results = text_to_image(text_embedding)
    output = [
        {'image_url': IMAGE_PATH_TEMPLATE.format(filename=img), 'similarity': float(score)}
        for img, score in results
    ]
    return jsonify(output)

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    image_id = data.get('imageId')
    feedback = data.get('feedback')
    user_input = data.get('userInput')

    if not image_id or feedback not in ['relevant', 'not_relevant']:
        return jsonify({'error': 'Invalid input'}), 400

    with open('feedback_log.txt', 'a') as f:
        f.write(f"{image_id},{feedback},{user_input}\n")
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)

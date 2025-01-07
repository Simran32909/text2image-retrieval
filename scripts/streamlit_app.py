import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import clip
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

ALIGNED_EMBEDDINGS_PATH = r"C:\Users\simra\PycharmProjects\cross_domain_transfer_learning\data\aligned_embeddings.npz"
CAPTIONS_PATH = r"C:\Users\simra\PycharmProjects\cross_domain_transfer_learning\data\captions_tokenized.csv"

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

st.title("Text-to-Image Retrieval App")

text_input = st.text_input("Enter a query (text):")

if text_input:
    suggestions = suggest_caption(text_input, captions)
    if suggestions:
        selected_caption = st.selectbox("Select a Caption", suggestions)
    else:
        st.write("No suggestions found.")

if st.button("Retrieve Images"):
    if text_input.strip():
        text_input = selected_caption if text_input.strip() == "" and 'selected_caption' in locals() else text_input.strip()
        text_embedding = model.encode_text(clip.tokenize([text_input]).to(device))

        results = text_to_image(text_embedding)
        st.subheader("Top Matching Images:")
        for img, score in results:
            image_path = f"C:/Users/simra/PycharmProjects/cross_domain_transfer_learning/data/images/flickr30k_images/{img}"
            st.image(image_path, caption=f"Similarity: {score:.4f}")
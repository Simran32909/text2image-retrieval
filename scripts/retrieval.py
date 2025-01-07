import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

ALIGNED_EMBEDDINGS_PATH = r"C:\Users\simra\PycharmProjects\cross_domain_transfer_learning\data\aligned_embeddings.npz"
CAPTIONS_PATH = r"C:\Users\simra\PycharmProjects\cross_domain_transfer_learning\data\captions_tokenized.csv"

aligned_data=np.load(ALIGNED_EMBEDDINGS_PATH)
text_emb=aligned_data["text"]
img_emb=aligned_data["image"]

metadata=pd.read_csv(CAPTIONS_PATH)
image_filenames = metadata["image"].tolist()
captions = metadata["clean_caption"].tolist()

def text_to_image(query_text_emb, top_k=5):
    similarities = cosine_similarity(query_text_emb.reshape(1, -1), img_emb).flatten()
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    results = [(image_filenames[i], similarities[i]) for i in top_k_indices]
    return results

def img_to_text(query_img_emb, top_k=5):
    similarities = cosine_similarity(query_img_emb.reshape(1, -1), text_emb).flatten()
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    results = [(captions[i], similarities[i]) for i in top_k_indices]
    return results

query_text_emb=text_emb[0]
text_to_image_res=text_to_image(query_text_emb)
print("Text-to-Image Retrieval Results:")
for img, score in text_to_image_res:
    print(f"Image: {img}, Similarity: {score:.4f}")

query_img_emb=img_emb[0]
img_to_text_res=img_to_text(query_img_emb)
print("Image-to-Text Retrieval Results:")
for txt, score in img_to_text_res:
    print(f"Caption: {txt}, Similarity: {score:.4f}")
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


ALIGNED_EMBEDDINGS_PATH = r"C:\Users\simra\PycharmProjects\cross_domain_transfer_learning\data\aligned_embeddings.npz"

aligned_data = np.load(ALIGNED_EMBEDDINGS_PATH)
text_embeddings = aligned_data["text"]
image_embeddings = aligned_data["image"]

similarities = cosine_similarity(image_embeddings, text_embeddings)

for i, sim in enumerate(similarities):
    best_match_idx = np.argmax(sim)
    print(f"Image {i} best matches Caption {best_match_idx} with similarity {sim[best_match_idx]:.4f}")

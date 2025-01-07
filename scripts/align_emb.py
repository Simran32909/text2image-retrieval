import numpy as np
import pandas as pd

TEXT_EMBEDDINGS_PATH = r"C:\Users\simra\PycharmProjects\cross_domain_transfer_learning\data\text_embeddings.csv"
IMAGE_EMBEDDINGS_PATH = r"C:\Users\simra\PycharmProjects\cross_domain_transfer_learning\data\image_embeddings.npy"
OUTPUT_ALIGNED_PATH = r"C:\Users\simra\PycharmProjects\cross_domain_transfer_learning\data\aligned_embeddings.npz"

text_embeddings_df = pd.read_csv(TEXT_EMBEDDINGS_PATH)
image_embeddings = np.load(IMAGE_EMBEDDINGS_PATH)

aligned_embeddings = []
TARGET_DIM = 512

def resize_embedding(embedding, target_dim):
    if len(embedding) > target_dim:
        return embedding[:target_dim]
    elif len(embedding) < target_dim:
        return np.pad(embedding, (0, target_dim - len(embedding)))
    return embedding

for idx, row in text_embeddings_df.iterrows():
    image_name = row["image"]
    text_embedding = np.fromstring(row["text_embedding"].strip("[]"), sep=",")
    text_embedding = resize_embedding(text_embedding, TARGET_DIM)

    image_idx = text_embeddings_df[text_embeddings_df["image"] == image_name].index[0]
    image_embedding = resize_embedding(image_embeddings[image_idx], TARGET_DIM)

    aligned_embeddings.append((text_embedding, image_embedding))

aligned_dict = {"text": [pair[0] for pair in aligned_embeddings],
                "image": [pair[1] for pair in aligned_embeddings]}
np.savez(OUTPUT_ALIGNED_PATH, text=np.array(aligned_dict["text"]), image=np.array(aligned_dict["image"]))
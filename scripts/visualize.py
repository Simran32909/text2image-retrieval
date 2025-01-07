import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

ALIGNED_EMBEDDINGS_PATH = r"C:\Users\simra\PycharmProjects\cross_domain_transfer_learning\data\aligned_embeddings.npz"
CAPTIONS_PATH = r"C:\Users\simra\PycharmProjects\cross_domain_transfer_learning\data\captions_tokenized.csv"

aligned_data = np.load(ALIGNED_EMBEDDINGS_PATH)
text_emb = aligned_data["text"]
img_emb = aligned_data["image"]
from sklearn.preprocessing import normalize
img_emb_normalized = normalize(img_emb)
text_emb_normalized = normalize(text_emb)
image_example = img_emb[0]
text_example = text_emb[0]

pca = PCA(n_components=2)
combined_emb = np.concatenate((img_emb, text_emb), axis=0)
reduced = pca.fit_transform(combined_emb)

plt.scatter(reduced[:len(img_emb), 0], reduced[:len(img_emb), 1], label='Image Embeddings')
plt.scatter(reduced[len(img_emb):, 0], reduced[len(img_emb):, 1], label='Text Embeddings')
plt.legend()
plt.show()
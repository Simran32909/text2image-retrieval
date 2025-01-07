import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm

CAPTIONS_PATH = r"C:\Users\simra\PycharmProjects\cross_domain_transfer_learning\data\captions_tokenized.csv"
OUTPUT_EMBEDDINGS_PATH = r"C:\Users\simra\PycharmProjects\cross_domain_transfer_learning\data\text_embeddings.csv"

df = pd.read_csv(CAPTIONS_PATH)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def generate_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

embeddings = []
for caption in tqdm(df["clean_caption"]):
    embedding = generate_embedding(caption)
    embeddings.append(embedding)

df["text_embedding"] = embeddings
df.to_csv(OUTPUT_EMBEDDINGS_PATH, index=False)

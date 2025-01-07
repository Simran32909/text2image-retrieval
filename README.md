# Text-to-Image Retrieval System

This project is a web application that enables users to retrieve images from a dataset by entering textual descriptions. The system leverages OpenAI's CLIP model to map textual input to image embeddings, enabling efficient and accurate text-to-image retrieval.

---

## How It Works

### 1. **Model and Dataset**
- The system uses the **CLIP (Contrastive Language–Image Pre-training)** model by OpenAI, which is designed to jointly learn image and text representations.
- **Image and Text Embeddings**: The pre-computed embeddings for the image dataset and captions are stored in a `.npz` file. These embeddings are generated using CLIP's `encode_image` and `encode_text` methods.
- The dataset includes:
    - A set of images with associated captions.
    - Preprocessed embeddings for efficient similarity computation.

### 2. **Text-to-Image Retrieval**
- When a user enters a text query:
    1. The text is tokenized and passed through the CLIP model to generate a query embedding.
    2. The cosine similarity between the query embedding and pre-computed image embeddings is calculated.
    3. The top `k` most similar images are returned, along with their similarity scores.

---

## Installation and Setup

### Prerequisites
- Python 3.1x
- Pip
- A compatible GPU for optimal performance

### Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/Simran32909/text2image-retrieval.git
   cd text2image-retrieval.git

2. **Set Up Virtual Environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install git+https://github.com/openai/CLIP.git

4. **Add Dataset**
- The dataset can be downloaded with the following Google Drive link: https://drive.google.com/file/d/1LmyHPaHKqN7oY5E7TqL0cFUBhdIdVb7e/view?usp=sharing
- Download the dataset locally and place it in the `data/` directory of the project:
    - `aligned_embeddings.npz`: Contains precomputed text and image embeddings.
    - `captions_tokenized.csv`: Metadata containing image captions.
- Place the images in the `static/images/` directory.

5. **Run Application**
   ```bash
   flask run

## Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request for any enhancements or bug fixes.

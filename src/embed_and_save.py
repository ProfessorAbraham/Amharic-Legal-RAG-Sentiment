import jsonlines
import numpy as np
from sentence_transformers import SentenceTransformer
import os

def load_chunks(path="data/processed/chunks.jsonl"):
    """Load chunked Amharic texts from JSONL."""
    texts = []
    ids = []
    with jsonlines.open(path) as reader:
        for item in reader:
            texts.append(item["text"])
            ids.append(item["id"])
    return texts, ids

def embed_texts(texts):
    """
    Generate embeddings using a multilingual model.
    Supports Amharic via xlm-r or similar models.
    """
    model = SentenceTransformer("sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens")
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

def save_embeddings(embeddings, ids, out_dir="data/processed"):
    """Save embeddings and metadata to disk."""
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/embeddings.npy", embeddings)

    # Save metadata (chunk IDs)
    with open(f"{out_dir}/metadata.txt", "w") as f:
        for idx, id in enumerate(ids):
            f.write(f"{id}\n")

if __name__ == "__main__":
    print("🧠 Loading chunks...")
    texts, ids = load_chunks()

    print("🧬 Generating embeddings...")
    embeddings = embed_texts(texts)

    print("💾 Saving embeddings...")
    save_embeddings(embeddings, ids)

    print("✅ Embedding process complete.")

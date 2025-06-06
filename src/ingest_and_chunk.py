import jsonlines
from typing import List

def load_documents(file_path):
    """Load Amharic legal documents from JSONL."""
    docs = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            docs.append(obj)
    return docs

def chunk_text(text: str, chunk_size=500, overlap=50) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def ingest_and_chunk(input_file="data/extracted_data (5) (2).jsonl", output_file="data/processed/chunks.jsonl"):
    docs = load_documents(input_file)
    chunked_data = []

    for doc in docs:
        doc_id = doc.get("id")
        content = doc.get("content", "")
        chunks = chunk_text(content)

        for i, chunk in enumerate(chunks):
            chunked_data.append({
                "id": f"{doc_id}_chunk_{i}",
                "text": chunk,
                "source_doc_id": doc_id
            })

    with jsonlines.open(output_file, mode='w') as writer:
        for item in chunked_data:
            writer.write(item)

if __name__ == "__main__":
    ingest_and_chunk()

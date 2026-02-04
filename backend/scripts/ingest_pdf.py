import os
import re
import json
from pypdf import PdfReader
from nltk.tokenize import sent_tokenize

PDF_DIR = "data/papers"
CHUNK_FILE = "data/processed_chunks.json"

WINDOW_SENTENCES = 6
STRIDE_SENTENCES = 3
MIN_CHARS = 300


def normalize_text(text: str) -> str:
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)

    return normalize_text(" ".join(pages))


def sliding_window_chunks(text: str):
    sentences = sent_tokenize(text)
    chunks = []

    i = 0
    while i < len(sentences):
        window = sentences[i:i + WINDOW_SENTENCES]
        chunk = " ".join(window)

        if len(chunk) >= MIN_CHARS:
            chunks.append(chunk)

        i += STRIDE_SENTENCES

    return chunks


def ingest_pdfs():
    all_chunks = []
    chunk_id = 0

    for filename in os.listdir(PDF_DIR):
        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(PDF_DIR, filename)
        print(f"[INGEST] Processing {filename}")

        text = extract_text_from_pdf(pdf_path)
        chunks = sliding_window_chunks(text)

        for chunk in chunks:
            all_chunks.append({
                "paper_id": filename,
                "chunk_id": f"{filename}_{chunk_id}",
                "text": chunk
            })
            chunk_id += 1

    with open(CHUNK_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"[INGEST] Saved {len(all_chunks)} chunks → {CHUNK_FILE}")


if __name__ == "__main__":
    ingest_pdfs()

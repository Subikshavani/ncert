import os
import json

EXTRACTED_ROOT = "extracted_text"

CHUNK_SIZE = 400   # words
OVERLAP = 50       # words


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start = end - overlap

    return chunks


def process_all_extracted_files():
    for root, _, files in os.walk(EXTRACTED_ROOT):
        for file in files:
            if not file.endswith(".json"):
                continue
            if file.endswith("_chunks.json"):
                continue

            file_path = os.path.join(root, file)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 🔒 SAFETY CHECK
            if "pages" not in data:
                continue

            chunks_output = {
                "class": data.get("class"),
                "subject": data.get("subject"),
                "language": data.get("language"),
                "source_pdf": data.get("source_pdf"),
                "chunks": []
            }

            chunk_id = 1

            for page in data["pages"]:
                text = page.get("text", "")
                page_no = page.get("page", -1)

                if not text or len(text.strip()) < 50:
                    continue

                text_chunks = chunk_text(text)

                for chunk in text_chunks:
                    chunks_output["chunks"].append({
                        "chunk_id": f"{chunks_output['source_pdf'].replace('.pdf','')}_{chunk_id}",
                        "text": chunk,
                        "page": page_no
                    })
                    chunk_id += 1

            if not chunks_output["chunks"]:
                continue

            output_file = file_path.replace(".json", "_chunks.json")

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(chunks_output, f, ensure_ascii=False, indent=2)

            print(f"✅ Created chunks: {output_file}")


if __name__ == "__main__":
    process_all_extracted_files()

import os
import json
import pdfplumber
import pytesseract
from PIL import Image

PDF_ROOT = "pdfs"
OUTPUT_ROOT = "extracted_text"


def extract_text_pdf(pdf_path):
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()

            if text and len(text.strip()) > 20:
                pages.append({
                    "page": i + 1,
                    "text": text
                })
            else:
                image = page.to_image(resolution=300).original
                ocr_text = pytesseract.image_to_string(image)
                pages.append({
                    "page": i + 1,
                    "text": ocr_text
                })

    return pages


def process_all_pdfs():
    print("🔍 Scanning PDFs...")
    print("📂 Contents of pdfs:", os.listdir(PDF_ROOT))

    for class_dir in os.listdir(PDF_ROOT):
        class_path = os.path.join(PDF_ROOT, class_dir)
        if not os.path.isdir(class_path):
            continue

        print(f"\n➡ Class: {class_dir}")

        for subject in os.listdir(class_path):
            subject_path = os.path.join(class_path, subject)
            if not os.path.isdir(subject_path):
                continue

            print(f"   📘 Subject: {subject}")

            for language in os.listdir(subject_path):
                language_path = os.path.join(subject_path, language)
                if not os.path.isdir(language_path):
                    continue

                print(f"      🌐 Language: {language}")
                files = os.listdir(language_path)
                print(f"         Files: {files}")

                for pdf_file in files:
                    if not pdf_file.lower().endswith(".pdf"):
                        continue

                    pdf_path = os.path.join(language_path, pdf_file)
                    print(f"         📄 Processing: {pdf_path}")

                    pages = extract_text_pdf(pdf_path)

                    output_dir = os.path.join(
                        OUTPUT_ROOT, class_dir, subject, language
                    )
                    os.makedirs(output_dir, exist_ok=True)

                    output_file = os.path.join(
                        output_dir,
                        pdf_file.replace(".pdf", ".json")
                    )

                    data = {
                        "class": class_dir.replace("class_", ""),
                        "subject": subject,
                        "language": language,
                        "source_pdf": pdf_file,
                        "pages": pages
                    }

                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)

                    print(f"         ✅ Saved: {output_file}")


if __name__ == "__main__":
    process_all_pdfs()

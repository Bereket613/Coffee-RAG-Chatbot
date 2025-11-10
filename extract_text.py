import os
import fitz
pdf_folder = 'data/pdf'
output_folder = 'data/text'

os.makedirs(output_folder, exist_ok=True)

for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        text_path = os.path.join(output_folder, pdf_file.replace('.pdf', '.txt'))
        with fitz.open(pdf_path) as doc:
            full_text = " "
            for page in doc:
                full_text += page.get_text()
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"Extracted text from {pdf_file} to {text_path}")            
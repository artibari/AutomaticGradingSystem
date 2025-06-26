# Automated Evaluation of Student Answers Using RAG

This project implements a Retrieval-Augmented Generation (RAG) based system to automatically evaluate descriptive and diagram-based student answers using Large Language Models (LLMs) and image-text similarity techniques.

## 📌 Features

- Automatic grading of both text and image-based answers
- Semantic text similarity using Sentence Transformers (GTE-base)
- Image similarity evaluation using CLIP
- RAG-based answer generation using LangChain and Mistral LLM
- Support for various question types: short, long, case-based, MCQs
- Excel-based input/output for integration and reporting
- CBSE Class 10 Science data aligned with NCERT textbooks

## 🔧 Technologies Used

- Python
- LangChain
- FAISS
- Mistral LLM via Ollama
- sentence-transformers (GTE-base)
- CLIP (openai/clip-vit-base-patch32)
- pdfplumber, paddleocr
- Pandas, OpenCV
- Streamlit (if UI is used)

## 📁 Folder Structure

project-root/
├── data/
│ ├── textbooks/ # NCERT PDFs
│ ├── diagrams/ # Extracted diagrams
│ └── excel_inputs/ # Student responses in Excel
├── embeddings/
├── models/
├── outputs/
├── app.py / main.py # Main processing script
├── utils/
│ └── grading_logic.py
│ └── diagram_processing.py
│ └── similarity_scoring.py
├── README.md
└── requirements.txt


## 🚀 How to Run

1. **Clone the repository:**
Install dependencies:
pip install -r requirements.txt

Run the main script:
python app.py
Provide the Excel input file with student answers in the correct format:
Columns: Q_No, Question_Type, Question, Student_Answer, Student_Image_Path, etc.
Check the output Excel file in the outputs/ folder containing:
Generated model answers
Similarity scores (F1, EM)
Final marks

📌 Future Enhancements
OCR for handwritten answer recognition
Multilingual support
Integration with LMS platforms
Support for other subjects beyond science
# Automated Evaluation of Text and Diagram Based Student Answers Using Retrieval-Augmented Generation
import os
import re
import fitz 
import pdfplumber
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from paddleocr import PaddleOCR
from IPython.display import Image as DisplayImage, display
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from evaluate import load

from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, util

###################################### To create vector dataset ##########################
def load_and_process_pdf_data(PDF_DIR, IMAGE_DIR):
    """
    Load and Process all PDFs.

    Args:
        PDF_DIR: Directory name where all PDF's are stored
        IMAGE_DIR: Directory name where all images are stored

    Returns:
        List of text documents and image documents.

    """
    text_docs = []
    image_docs = []
    for filename in os.listdir(PDF_DIR):
        if not filename.endswith(".pdf"):
            continue

        pdf_path = os.path.join(PDF_DIR, filename)
        text_area_threshold = 0.2
        with pdfplumber.open(pdf_path) as pdf:
            doc_fitz = fitz.open(pdf_path)
            for i, (pdfplumber_page, fitz_page) in enumerate(zip(pdf.pages, doc_fitz)):
                page_key = f"{filename}_page_{i+1}"
                text = pdfplumber_page.extract_text() or ""
                text_docs.append(Document(
                    page_content=text,
                    metadata={"source": page_key}
                ))

                zoom_factor = 2  # 144 DPI rendering
                matrix = fitz.Matrix(zoom_factor, zoom_factor)
                pix = fitz_page.get_pixmap(matrix=matrix)

                img = np.frombuffer(pix.samples, dtype=np.uint8)
                height, width = pix.height, pix.width
                num_channels = pix.n

                if len(img) != height * width * num_channels:
                    print(f"Array size mismatch on page {i + 1}. Skipping.")
                    continue

                img = img.reshape(height, width, num_channels)

                if num_channels == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                elif num_channels == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                thresh = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
                )

                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for j, cnt in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w < 150 or h < 150:
                        continue

                    padding_vertical = 70   # Top and Bottom
                    padding_horizontal = 100  # Left and Right
                    
                    x1 = max(0, x - padding_horizontal)
                    y1 = max(0, y - padding_vertical)
                    x2 = min(img.shape[1], x + w + padding_horizontal)
                    y2 = min(img.shape[0], y + h + padding_vertical)
                    
                    diagram = img[y1:y2, x1:x2]

                    img_name = f"{page_key}_diagram{j}.png"
                    diagram_path = os.path.join(IMAGE_DIR, img_name)
                    cv2.imwrite(diagram_path, diagram)

                    try:
                        image = Image.open(diagram_path).convert("RGB")
                        width, height = image.size

                        # OCR text area ratio (skip dense text blocks)
                        result = ocr_model.ocr(diagram_path, cls=True)
                        total_text_area = sum(
                            max(abs(x2 - x1), abs(x3 - x4)) * max(abs(y3 - y2), abs(y1 - y4))
                            for line in result[0]
                            for (x1, y1), (x2, y2), (x3, y3), (x4, y4) in [line[0]]
                        )
                        text_ratio = total_text_area / (width * height)

                        if text_ratio > text_area_threshold:
                            print(f"Skipping text-heavy image: {diagram_path}")
                            os.remove(diagram_path)
                            continue

                        # ---  Crop full-width region below the image ---
                        caption_text = ""
                        caption_y_start = y + h // 2
                        caption_y_end = img.shape[0]
                        x1 = 0
                        x2 = img.shape[1]

                        caption_crop = img[caption_y_start:caption_y_end, x1:x2]
                        caption_temp_path = os.path.join(IMAGE_DIR, f"{page_key}_caption{j}.png")
                        cv2.imwrite(caption_temp_path, caption_crop)
                        try:
                            ocr_result = ocr_model.ocr(caption_temp_path, cls=True)
                            lines_with_coords = ocr_result[0]

                            figure_line_found = False
                            caption_lines = []
                            base_x = None
                            base_y = None
                            idx = -1  # Default value in case "Figure" isn't found

                            # Step 1: Find the "Figure"/"Fig." line
                            for i, (box, (text, _)) in enumerate(lines_with_coords):
                                if re.search(r'\bFigure\b', text, re.IGNORECASE):
                                    x_coords = [p[0] for p in box]
                                    y_coords = [p[1] for p in box]
                                    base_x = min(x_coords)
                                    base_y = max(y_coords)
                                    figure_line_found = True
                                    idx = i
                                    # Only append if line is clean (≤ 2 spaces)
                                    if not re.search(r"\s{3,}", text):
                                        caption_lines.append(text.strip())
                                    break  # Found "Figure", move to next lines

                            # Step 2: Collect lines after "Figure" up to 6 lines
                            if figure_line_found:
                                line_count = len(caption_lines)
                                for next_box, (next_text, _) in lines_with_coords[idx+1:]:
                                    if line_count >= 6:
                                        break

                                    #Stop if another figure is detected — prevent capturing second figure
                                    if re.search(r'\bFigure\b', next_text, re.IGNORECASE):
                                        break

                                    next_x_coords = [p[0] for p in next_box]
                                    next_y_coords = [p[1] for p in next_box]
                                    next_x_min = min(next_x_coords)
                                    next_y_top = min(next_y_coords)

                                    # Must be vertically below and horizontally aligned
                                    if abs(next_x_min - base_x) > 40 or next_y_top < base_y:
                                        continue

                                    # Skip line if it has more than 2 consecutive spaces
                                    if re.search(r"\s{3,}", next_text):
                                        continue

                                    caption_lines.append(next_text.strip())
                                    base_y = max(next_y_coords)  # update Y to support descending order
                                    line_count += 1

                            caption_text = " ".join(caption_lines).strip()

                        except Exception as e:
                            print(f"OCR caption error: {e}")

                        finally:
                            if os.path.exists(caption_temp_path):
                                os.remove(caption_temp_path)

                        # Store as Document
                        image_docs.append(Document(
                            page_content=caption_text,
                            metadata={
                                "img_path": diagram_path,
                                "source": page_key,
                            }
                        ))

                    except Exception as e:
                        print(f"Failed image/caption handling for {diagram_path}: {e}")

    return text_docs, image_docs

def clean_text(text):
    """
    Cleans text for RAG purposes.

    Args:
        text: text from text documents 
        
    Returns:
        Cleaned text
    """
    # Lowercase
    text = text.lower()
    # Replace \n with space
    text = text.replace("\n", " ")
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()#     
    # Collapse repeated characters: "ffffiiigggg" -> "fig"
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    # Remove Headers/Footers and "Reprint" lines
    text = re.sub(r"^\s*“.*Fischer.*$", "", text, flags=re.MULTILINE) #Remove quote at the beginning
    text = re.sub(r"Reprint \d{4}-\d{2}", "", text)
    text = re.sub(r"CHAPTER\s*\d+", "", text) #Remove chapter headings
    text = re.sub(r"Q U E S T I O N S", "", text) #Remove questions headings
    text = re.sub(r"AAAAAccccctttttiiiiivvvvviiiiitttttyyyyy\s*\d+.\d+", "", text) #Remove Activity headings
    text = re.sub(r"Chemical Reactions and Equations\s*\d+", "", text) #Remove Chemical Reactions and Equations headings
    text = re.sub(r"\f", "", text) #Remove page breaks
    text = re.sub(r"Our Environment\s*\d+", "", text) #Remove Our Environment headings
    # Remove Figure References (without the actual figures)
    text = re.sub(r"Figure\s+[X\d.]+", "", text)
    # Remove specific phrases
    text = re.sub(r"Consider the following situations of daily life and think what happens\s*when –", "", text)
    text = re.sub(r"In all the above situations, the nature and the identity of the initial", "", text)
    # Handle Numbered Lists (remove numbers, but keep the content)
    text = re.sub(r"^\s*\d+\.\s*", "", text, flags=re.MULTILINE) #Remove numbered list prefixes
    # Remove extra whitespace and line breaks
    text = re.sub(r"\s+", " ", text) #Replace multiple spaces with single space
    text = re.sub(r"[\r\n]+", "\n", text) #Normalize line breaks
    text = text.strip() #Remove leading/trailing whitespace
    return text

def create_chunk_docs(text_docs, image_docs):
    """
    Splitting text and create chunk documents.

    Args:
        text_docs: Text documents 
        image_docs: Image documents
        
    Returns:
        chunk documents
    """
    # Text splitter config (Markdown-friendly)
    MARKDOWN_SEPARATORS = [
        "\n#{1,6} ",
        "```\n",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    chunked_docs = []

    combined_docs = text_docs + image_docs

    for doc in combined_docs:
        chunks = text_splitter.split_text(doc.page_content)
        cleaned_chunks = [clean_text(chunk) for chunk in chunks]
        
        for i, chunk in enumerate(cleaned_chunks):
            chunked_docs.append(Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    "chunk_index": i,
                    "type": "image" if "img_path" in doc.metadata else "text"
                }
            ))
    return chunked_docs

###################################### Data Preprocessing End ########################


def get_model_answer(answer_word_length, optimized_question, combined_vectorstore, llm):
    """
     Get the model answers from text and image documents.

    Args:
        answer_word_length(int): Length of words of answers
        optimized_question: Original question which is optimized
        combined_vectorstore: Name of vectorestore which stored text and image documents
        llm:Large language model name
        
    Returns:
        Model answer and image path name
    """ 

    # Prompt Template
    prompt_template = f"""
    You are a science student writing an exam answer. Based strictly on the provided context from physics, chemistry, or biology, answer the question below.
 
    **Step 1: Identify the Question Type**
    Before answering, determine if the question is:
    - Numerical (involving calculations)
    - Theoretical (descriptive)
    - Objective (multiple choice or true/false)
    - Diagram-based (asks to draw, label, or show a figure)
 
    **Step 2: Follow Only the Instructions Matching the Identified Type**
 
    **For NUMERICAL questions:**
    - Do NOT write theory or background.
    - Write only the solution steps:
        1. List known values (e.g., i = ..., t = ...)
        2. Convert units (if needed)
        3. Write the correct formula
        4. Substitute the values
        5. Perform calculations
        6. Final answer with proper units
 
    **For THEORETICAL questions:**
    - Write clear, brief sentences. No long paragraphs.
 
    **For OBJECTIVE questions (MCQ, true/false):**
    - Choose the correct answer **only from the options provided** in the question.
    - Do NOT invent new choices.
    - Do NOT answer if the context doesn't clearly support one of the options.
 
    **For DIAGRAM questions:**
    - Describe the required diagram (title, parts, labels)
    - Then retrieve or generate it
 
    If the answer is not present in the context, reply:
    "The context does not contain this information."
 
    **Limit your answer to {answer_word_length} words or fewer.**
 
    ---
 
    Context:
    {{context}}
 
    Question:
    {{question}}
 
    Answer:
    """
 
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question","answer_word_length"]
    )

    chain_type_kwargs = {"prompt": PROMPT}

    # Set up retriever
    retriever = combined_vectorstore.as_retriever(search_kwargs={"k": 3})

    # RAG QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )

    # Run query
    result = qa_chain.invoke({"query": optimized_question})
    answer_text = result['result']

    # Diagram logic
    image_keywords = ["draw", "show", "diagram", "visual"]
    image_paths = []

    if any(kw in optimized_question.lower() for kw in image_keywords):
        for doc in result['source_documents']:
            img_path = doc.metadata.get('img_path')
            if img_path and os.path.exists(img_path):
                image_paths.append(img_path)

        # Display if any images found
        if image_paths:
            print(answer_text)
            for path in image_paths:
                display(DisplayImage(filename=path))
        else:
            print(answer_text)
            print("(Diagram requested but no image found in sources.)")
    else:
        print(answer_text)

    return {
        "answer": answer_text,
        "image_paths": image_paths
    }


def optimize_question(original_question, llm):
    """
     Rewrite the following question for optimal information retrieval.

    Args:
        original question(str):Original question asked by user 
        llm:Large language model name
        
    Returns:
        Optimize question text
    """     
    # Define a query optimization prompt
    query_optimization_prompt = f"""
    You are a helpful assistant simplifying science exam questions for better retrieval from a textbook.

    Your goal is to make the question **clear and more concise** while **preserving its original meaning, structure, and intent**. 

    - Do **not** split the question into sub-questions.
    - Do **not** add, remove, or change any factual details, values, units, or options.
    - Do **not** enrich the question with additional context or assumptions.
    - If the original question includes terms like **"draw"**, **"diagram"**, **"visual"**, or **"show"**, retain those exactly as they are, since they indicate a visual answer.
    
    Original Question:
    {original_question}

    Optimized Query:

    Simplified Question (same meaning, no new info):
    """

    # Use the LLM to optimize the query
    optimized_question = llm.invoke(query_optimization_prompt)

    print(f"Original Query: {original_question}")
    print(f"Optimized Query: {optimized_question}")
    return optimized_question

def similarity_matching(student_answer, model_answer):
    """
        Check the similarity matching between student answer and model answer. 
    Args:
        student_answer:Student answer text
        model_answer:Text generated by model
        
    Returns:
        similarity score(float)
    """     
    emb1 = text_model.encode(model_answer,convert_to_tensor=True)
    emb2 = text_model.encode(student_answer,convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2)
    print(f'Text Similarity Score: {similarity.item():.4f}')
    return similarity.item()

def compute_image_similarity(image_path1, image_path2):
    """
        Check the image similarity matching between student answer image and model answer image 
    Args:
        image_path1:Model amswer image
        image_path2:Student answer image
        
    Returns:
        similarity score(float)
    """     
    image1 = Image.open(image_path1).convert("RGB")
    image2 = Image.open(image_path2).convert("RGB")
    inputs = clip_processor(images=[image1, image2], return_tensors="pt", padding=True)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    similarity = torch.matmul(image_features, image_features.T)
    return similarity[0, 1].item()


def get_qa_model_eval(prediction_result, ground_result):
    """
     Calculate model evaluation result.
    Args:
        prediction_result:Prediction text
        ground_result:student answers text
        
    Returns:
        Evaluation result of F1 and exact match
    """     
    # Load F1/EM scoring evaluator
    qa_eval = load("squad")
    eval_result = qa_eval.compute(predictions=[{"id": "1", "prediction_text": prediction_result}],
                              references=[{"id": "1", "answers": {"text": [ground_result], "answer_start": [0]}}])
    print("eval_result--",eval_result)
    return eval_result["f1"], eval_result["exact_match"]


def get_marking_scheme(question_num):
    """
       Define the marking scheme according to answer type
    Args:
        question_num:question number
                
    Returns:
        marks(int)

    """     
    q_num = int(question_num)

    if 21 <= q_num <= 26:
        return 30, 50, 2  # Very short answer
    elif 27 <= q_num <= 33:
        return 50, 80, 3  # Short answer
    elif 34 <= q_num <= 36:
        return 80, 120, 5  # Long answer
    elif 37 <= q_num <= 39:
        return 80, 120, 4  # Case/source-based
    else:
        return 1, 30, 1  # Multiple choice


def grade_to_answer(student_answer, text_similarity, diagram_required=False, image_similarity=None,
                    text_weight=0.7, image_weight=0.3,min_words=int, max_words=int, max_marks=int):
    """
        Get the final grade for student answer
    Args:
        student_answer(str):Student answer text
        text_similarity(float):Text similarity score
        diagram_required(boolean):If diagram required mention True else false
        image_similarity:Image similarity score
        text_weight(float):float value
        image_weight(float):float value
        min_words(int):Minimum length of words
        max_words(int):Maximum length of words
        max_marks(int): Maximum marks  

    Returns:
        marks(int)
    """     
    if diagram_required and image_similarity is not None:
        final_similarity = (text_weight * text_similarity) + (image_weight * image_similarity)
    else:
        final_similarity = text_similarity

    word_count = len(student_answer.split())
    marks = 0

    # Determine base score from similarity
    if final_similarity > 0.85:
        base_score = max_marks
    elif final_similarity > 0.6:
        base_score = max_marks * 0.5
    else:
        if max_marks==1:
            base_score=0
        else:
            base_score = max_marks * 0.25

    # Apply penalty if word count is below min
    if word_count < min_words:
        ratio = word_count / min_words
        penalty_score = base_score * ratio
        marks = round(penalty_score, 1)
    else:
        marks = round(base_score, 1)

    # Optional: Cap marks if word count is far beyond max (no penalty)
    if word_count > max_words:
        print(f"Note: Word count exceeds {max_words} words. Suggest making answers concise.")

    print(f"Final Similarity Score: {final_similarity:.4f}, Word Count: {word_count}, Marks Awarded: {marks}")
    return marks

def process_questions_from_excel(file_path, combined_vectorstore, llm):
    """
       Read and update the excel file
    Args:
        file_path:Excel sheet file path name
        combined_vectorstore: Combined Text and image documents directory name
        llm:Large language model name
                
    Returns:
        None

    """  
    is_diagram=False
    image_score = None
    df = pd.read_excel(file_path)
    # loop over each row
    for index, row in df.iterrows():
        original_question = row["Questions"]
        q_type = row["Question_type"]

        if q_type == "Multiple_choice_question":
            answer_word_length = 30
            result = get_model_answer(answer_word_length, original_question, combined_vectorstore, llm)
        elif q_type == "Very_short_answer_type_question":
            answer_word_length = 50
            optimized_question = optimize_question(original_question, llm)
            result = get_model_answer(answer_word_length, optimized_question, combined_vectorstore, llm)    
        elif q_type == "Short_answer_type_question":
            answer_word_length = 80
            optimized_question = optimize_question(original_question, llm)
            result = get_model_answer(answer_word_length, optimized_question, combined_vectorstore, llm)
        elif q_type == "long_answer_type_question":
            answer_word_length = 120
            optimized_question = optimize_question(original_question, llm)
            result = get_model_answer(answer_word_length, optimized_question, combined_vectorstore, llm)
        
        # model evaluation
        eval_f1, eval_em = get_qa_model_eval(result["answer"], row["Student_Answer"])

        # Answer evaluation
        min_words, max_words, max_marks = get_marking_scheme(row["Q_No"])
        s_score = similarity_matching(result["answer"], row["Student_Answer"])
            
        if result["image_paths"] and pd.notna(row["Student_Image_path"]) and row["Student_Image_path"] != "":
            image_score = compute_image_similarity(result["image_paths"][0], row["Student_Image_path"])
            is_diagram = True
            
        marks = grade_to_answer(row["Student_Answer"], s_score, is_diagram, image_score, min_words=min_words, max_words=max_words, max_marks=max_marks)

        # Save the answer to the DataFrame
        df.at[index, 'Model_answers'] = result["answer"]
        df.at[index, 'Model_Image_path'] = ', '.join(result["image_paths"]) if result["image_paths"] else ''
        df.at[index, 'Eval_F1'] = eval_f1
        df.at[index, 'Eval_em'] = eval_em
        df.at[index, 'Marks'] = marks

    # Save the updated Excel
    df.to_excel(file_path, index=False)
    print(f"Processing complete. Saved to: {file_path}")

if __name__ == "__main__":
    # Initialize OCR once (outside loop)
    ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

    # Question sheet file path. Columns name = "Questions",	"Question_type",	"Model_answers",	"Image_paths", "Student_answers"
    file_path = "one_que_ans.xlsx"

    # Embedding Model
    embedder = HuggingFaceEmbeddings(model_name="clip-ViT-B-32")

    # Load vector data from Local
    combined_vectorstore = FAISS.load_local("faiss_vector_store", embedder, allow_dangerous_deserialization=True)
    
    # Setup Ollama LLM
    llm = OllamaLLM(model="Mistral:latest",
                    temperature=0.3,
                    max_tokens=1000 )
   
    # Initialize answer Evaluation models 
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    # Final run
    process_questions_from_excel(file_path, combined_vectorstore, llm)
    




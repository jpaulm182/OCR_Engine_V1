import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from pdf2image import convert_from_path
from pdf_processor import extract_text_from_pdf
from ocr_engine import ocr_image
from image_preprocessor import deskew_image, remove_lines, detect_and_remove_boxes
from post_processor import correct_ocr_errors, remove_unwanted_characters
from skimage.color import rgb2gray

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def convert_pdf_to_images(pdf_path, output_dir, dpi=350):
    images = convert_from_path(pdf_path, dpi=dpi)
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f'image_{i}.png')
        image.save(image_path, 'PNG')
        image_paths.append(image_path)
    return image_paths

def preprocess_image(image_path): 
    # Adjust path to normal format for the operating system
    image_path = os.path.normpath(image_path) 
    processed_image_path = deskew_image(image_path)
    processed_image_path = detect_and_remove_boxes(processed_image_path)
    processed_image_path = remove_lines(processed_image_path,200, 100,200, 20)
    processed_image_path = remove_lines(processed_image_path, 200, 75,50, 20)
    
    print(f"Processed image path3: {processed_image_path}")
    return processed_image_path

def postprocess_ocr_result(ocr_result):
    corrected_text = correct_ocr_errors(ocr_result)
    cleaned_text = remove_unwanted_characters(corrected_text)
    return cleaned_text

def save_extracted_text(extracted_text, pdf_path):
    with open(f"{os.path.splitext(pdf_path)[0]}_extracted.txt", "w") as text_file:
        text_file.write(''.join(extracted_text))

def save_cleaned_text(cleaned_text, image_path, pdf_path):
    with open(f"{os.path.splitext(image_path)[0]}.txt", "w") as text_file:
        text_file.write(cleaned_text)
    with open(f"{os.path.splitext(pdf_path)[0]}_compiled.txt", "a") as compiled_file:
        compiled_file.write(cleaned_text)

def process_pdf_document(pdf_path):
    print(f"Processing file: {pdf_path}")
    output_dir = os.path.join(os.path.dirname(pdf_path), 'image_output')
    ensure_directory_exists(output_dir)
    extracted_text = extract_text_from_pdf(pdf_path)
    image_paths = convert_pdf_to_images(pdf_path, output_dir)
    for image_path in image_paths:
        processed_image_path = preprocess_image(image_path)        
        ocr_result = ocr_image(processed_image_path)
        cleaned_text = postprocess_ocr_result(ocr_result)
        save_extracted_text(extracted_text, pdf_path)
        save_cleaned_text(cleaned_text, image_path, pdf_path)
    print("All files processed successfully")

if __name__ == "__main__":
    selected_pdf = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if selected_pdf:
        process_pdf_document(selected_pdf)
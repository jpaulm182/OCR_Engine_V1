import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from pdf2image import convert_from_path
from pdf_processor import extract_text_from_pdf
from ocr_engine import ocr_image
from image_preprocessor import deskew_image, get_text_regions, remove_lines_and_boxes
from post_processor import correct_ocr_errors, remove_unwanted_characters

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
    processed_image_path = deskew_image(image_path)
    print(f"Processed image path: {processed_image_path}")
    #save the processed image to a file in the same directory as the image
    processed_image_path = os.path.splitext(image_path)[0] + '_deskewed.png'
    
    processed_image_path = remove_lines_and_boxes(processed_image_path, left_margin=200,minLineLength=100, maxLineGap=15)
    
    processed_image_path = remove_lines_and_boxes(processed_image_path, left_margin=200, minLineLength=20, maxLineGap=20)
   
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
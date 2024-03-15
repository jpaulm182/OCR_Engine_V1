import os
from concurrent.futures import ProcessPoolExecutor
from pdf2image import convert_from_path
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from pdf2image import convert_from_path
from pdf_processor import extract_text_from_pdf
from ocr_engine import ocr_image
from image_preprocessor import deskew_image, remove_lines, threshold_image
from post_processor import correct_ocr_errors, remove_unwanted_characters
from skimage.color import rgb2gray
import concurrent.futures
import multiprocessing
import time
from PyPDF2 import PdfFileReader
from pdf2image import convert_from_path

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_image(image, image_path):
    image.save(image_path, 'PNG')
    return image_path

def convert_pdf_to_images_parallel(pdf_path, output_dir, dpi=350):
    images = convert_from_path(pdf_path, dpi=dpi)
    image_paths = []

    with ProcessPoolExecutor() as executor:
        for i, image in enumerate(images):
            image_path = os.path.join(output_dir, f'image_{i}.png')
            future = executor.submit(save_image, image, image_path)
            image_paths.append(future.result())

    return image_paths

def preprocess_image(image_path): 
    # Adjust path to normal format for the operating system
    image_path = os.path.normpath(image_path) 
    processed_image_path = threshold_image(image_path, threshold=0.4)
    #processed_image_path = binarize_image(processed_image_path)
    processed_image_path = deskew_image(processed_image_path)   
    processed_image_path = remove_lines(processed_image_path,left_margin=200, min_width=0, min_height=0, max_width= 3500, max_height=200)
    processed_image_path = deskew_image(processed_image_path)
    print(f"Processed image path3: {processed_image_path}")
    return processed_image_path

def preprocess_images_in_parallel(image_paths):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        processed_image_paths = list(executor.map(preprocess_image, image_paths))
    return processed_image_paths
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

def process_image(args):
    image_path, pdf_path = args
    processed_image_path = preprocess_image(image_path)        
    ocr_result = ocr_image(processed_image_path)
    cleaned_text = postprocess_ocr_result(ocr_result)
    save_cleaned_text(cleaned_text, image_path, pdf_path)
    return cleaned_text

def get_num_pages(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf = PdfFileReader(file)
        return pdf.getNumPages()

def convert_page_to_image(args):
    page, pdf_path, output_dir = args
    images = convert_from_path(pdf_path, first_page=page, last_page=page)
    image_path = f"{output_dir}/image_{page}.png"
    images[0].save(image_path, 'PNG')
    return image_path

def convert_pdf_to_images_parallel(pdf_path, output_dir):
    num_pages = get_num_pages(pdf_path)
    with multiprocessing.Pool() as pool:
        image_paths = pool.map(convert_page_to_image, [(i, pdf_path, output_dir) for i in range(1, num_pages + 1)])
    return image_paths

def process_pdf_document(pdf_path):
    start_time = time.time()
    print(f"Processing file: {pdf_path}")
    output_dir = os.path.join(os.path.dirname(pdf_path), 'image_output')
    ensure_directory_exists(output_dir)
    extracted_text = extract_text_from_pdf(pdf_path)
    image_paths = convert_pdf_to_images_parallel(pdf_path, output_dir)
    # Create a list of tuples to pass to pool.map
    args = [(image_path, pdf_path) for image_path in image_paths]
    with multiprocessing.Pool() as pool:
        cleaned_texts = pool.map(process_image, args)

    save_extracted_text(extracted_text, pdf_path)
    end_time = time.time()
    print("All files processed successfully")
    print(f"Time taken: {end_time - start_time} seconds")
    
if __name__ == "__main__":
    selected_pdf = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if selected_pdf:
        process_pdf_document(selected_pdf)
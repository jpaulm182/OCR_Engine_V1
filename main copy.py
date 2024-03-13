from pdf_processor import extract_text_from_pdf
from ocr_engine import ocr_image
from image_preprocessor import apply_thresholding, remove_noise,deskew_image, get_text_regions, remove_lines_and_boxes
from PIL import Image

#from layout_analyzer import find_contours, analyze_layout, remove_lines_and_boxes
from post_processor import correct_ocr_errors, remove_unwanted_characters
import tkinter as tk
from tkinter import filedialog
import os
from pdf2image import convert_from_path
from tkinter import messagebox

def extract_text_and_images_from_pdf(pdf_path, output_dir):
    """
    Extract text and images from all pages of the PDF.

    :param pdf_path: Path to the PDF document to be processed
    :param output_dir: Directory to save the extracted images
    :return: Extracted text and list of image paths
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert PDF pages to images
    images = convert_from_path(pdf_path, dpi = 350) #parameter 'dpi' can be used to set the resolution of the images
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f'image_{i}.png')
        image.save(image_path, 'PNG')
        image_paths.append(image_path)

    # Extract text from the PDF
    extracted_text = extract_text_from_pdf(pdf_path)
    return extracted_text, image_paths

def process_pdf_document(pdf_path, actions_order):
    actions = {
        "remove_lines": remove_lines_and_boxes,
        "deskew": deskew_image,
        "ocr": ocr_image,
        "correct_ocr_errors": correct_ocr_errors,
        "remove_unwanted_characters": remove_unwanted_characters
    }

    print(f"Processing file: {pdf_path}")
    output_dir = os.path.join(os.path.dirname(pdf_path), 'image_output')
    extracted_text, image_paths = extract_text_and_images_from_pdf(pdf_path, output_dir)

    for image_path in image_paths:
        processed_image_path = image_path
        for action in actions_order:
            processed_image_path = actions[action](processed_image_path)

        # Save the results to a file using the same name as the image file then create a compiled file with all the results
        with open(f"{os.path.splitext(image_path)[0]}.txt", "w") as text_file:
            text_file.write(processed_image_path)
        # Now combine all the results into a single file
        with open(f"{os.path.splitext(pdf_path)[0]}_compiled.txt", "a") as compiled_file:
            compiled_file.write(processed_image_path)
            
    print("All files processed successfully")

def move_up():
    selected = listbox.curselection()
    if not selected:
        return
    for index in selected:
        if index == 0:
            continue
        listbox.insert(index-1, listbox.get(index))
        listbox.delete(index+1)

def move_down():
    selected = listbox.curselection()
    if not selected:
        return
    for index in selected[::-1]:
        if index == listbox.size() - 1:
            continue
        listbox.insert(index+2, listbox.get(index))
        listbox.delete(index)

def start_processing():
    actions_order = list(listbox.get(0, tk.END))
    try:
        process_pdf_document(pdf_path, actions_order)
        messagebox.showinfo("Success", "All files processed successfully")
    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()

listbox = tk.Listbox(root)
listbox.pack(pady=10)

for action in ["grayscale", "threshold", "denoise", "remove_lines", "deskew", "ocr", "correct_ocr_errors", "remove_unwanted_characters"]:
    listbox.insert(tk.END, action)

up_button = tk.Button(root, text="Move Up", command=move_up)
up_button.pack()

down_button = tk.Button(root, text="Move Down", command=move_down)
down_button.pack()

start_button = tk.Button(root, text="Start Processing", command=start_processing)
start_button.pack(pady=10)

root.mainloop()
if __name__ == "__main__":
    selected_pdf = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if selected_pdf:
        process_pdf_document(selected_pdf)
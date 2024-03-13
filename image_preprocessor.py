from PIL import Image, ImageFilter, ImageOps
import cv2
import numpy as np
import pytesseract
import os
from skimage import io
from skimage.transform import rotate
from skimage import filters

from skimage.util import img_as_ubyte
from skimage.color import rgb2gray, rgba2rgb
from skimage.filters import threshold_otsu
from skimage.filters import median
from skimage.morphology import disk

def deskew_image(image_path):
    #the image must be 3 channel (RGB) for the deskew function to work
    image = io.imread(image_path)
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = rgba2rgb(image)
    # Convert the image to grayscale
    if len(image.shape) == 3:  # Check if the image is not grayscale
        gray = rgb2gray(image)
    else:
        gray = image  # If the image is grayscale, use it as is

    # Apply noise reduction
    selem = disk(1)
    noise_reduced = median(gray, selem) 

    # Apply binarization
    thresh = threshold_otsu(noise_reduced) #thresholding using otsu's method which is a global thresholding method that assumes a bimodal distribution of pixel intensities because it is a good way to separate the background from the foreground
    binary = noise_reduced > thresh #binary is a boolean array where True is the foreground and False is the background

    '''# Apply edge detection
    edges = cv2.Canny(img_as_ubyte(binary), 100, 255, apertureSize=3)
    #invert image
    edges = cv2.bitwise_not(edges)
    '''    
    # Apply deskew
    angle = filters.try_all_threshold(binary, figsize=(10, 8))
    print(angle) #try_all_threshold is a function that tries all the thresholding methods available in scikit-image
    #return the path of the deskewed image
    #save the deskewed image to a file in the same directory as the image
    deskewed_path = os.path.splitext(image_path)[0] + '_deskewed.png'
    #conert the image to a PIL image
    deskewed = Image.fromarray(binary)
    deskewed.save(deskewed_path)

    return deskewed_path

def get_text_regions(image_path: str):
    #get blocks of text regions from the image to be used as a mask
    #use tesseract
    image = Image.open(image_path)
    text = pytesseract.image_to_boxes(image)
    text_regions = []
    for line in text.split('\n'):
        if line:
            _, x, y, w, h, _ = line.split(' ')
            text_regions.append((int(x), int(y), int(w), int(h)))
    return text_regions

def load_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

def create_mask(gray, text_regions):
    mask = np.zeros_like(gray)
    for region in text_regions:
        x, y, w, h = region
        mask[y:y+h, x:x+w] = 255
    return mask

def erase_left_margin(image, left_margin):
    image[:, :left_margin] = 255
    return image

def detect_lines(gray, mask, minLineLength, maxLineGap):
    '''edges = cv2.Canny(gray, 50, 150, apertureSize=3) #apertureSize is the size of the Sobel kernel used to find image gradients'''
    lines = cv2.HoughLinesP(gray, 1, np.pi/180, 100, minLineLength=100, maxLineGap=15 )
    return lines

def remove_lines(image, lines, mask):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if mask[y1, x1] == 0 and mask[y2, x2] == 0:  # Only remove lines outside text regions
                cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 5)
    return image

def detect_and_remove_boxes(image, mask):
    #convert the image to CV_8UC1
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) > 1000 and mask[y + h//2, x + w//2] == 0:  # Large area and outside text regions
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)
    return image

def save_image(image, image_path):
    result = Image.fromarray(image)
    result_path = os.path.splitext(image_path)[0] + '_cleaned.png'
    result.save(result_path)
    return result_path

def remove_lines_and_boxes(image_path, left_margin, minLineLength, maxLineGap):
    text_regions = get_text_regions(image_path=image_path)
    image, gray = load_image(image_path=image_path)
    mask = create_mask(gray=gray, text_regions=text_regions)
    image = erase_left_margin(image=image, left_margin=left_margin)
    lines = detect_lines(gray=gray, mask=mask, minLineLength=minLineLength, maxLineGap=maxLineGap)
    image = remove_lines(image=image, lines=lines, mask=mask)
    image = detect_and_remove_boxes(image=image, mask=mask)
    result_path = save_image(image=image, image_path=image_path)
    #save the mask to a file in the same directory as the image
    mask_path = os.path.splitext(image_path)[0] + '_mask.png'
    io.imsave(mask_path, mask)
    #save the result to a file in the same directory as the image
    result = Image.fromarray(image)
    result_path = os.path.splitext(image_path)[0] + '_cleaned.png'
    result.save(result_path)
    
    return result_path
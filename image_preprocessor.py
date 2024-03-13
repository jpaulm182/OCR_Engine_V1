import cv2
import numpy as np
import os
import pytesseract

from PIL import Image, ImageFilter, ImageOps
from imutils import grab_contours
from skimage import io, transform, util, color, morphology
from skimage.filters import threshold_otsu, rank
from skimage.filters import median
from skimage.morphology import disk
from skimage.color import rgba2rgb, rgb2gray
from scipy.ndimage import interpolation as inter

def calculate_skew(image):
    # convert to binary
    image = cv2.bitwise_not(image)
    threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # calculate the moments to estimate the skew angle
    m = cv2.moments(threshold)
    if abs(m['mu02']) < 1e-2:
        return 0
    skew = m['mu11']/m['mu02']
    return skew

def threshold_image(image_path):
    image = io.imread(image_path)
    #convert to grayscale
    
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = rgba2rgb(image)
    gray = rgb2gray(image)
    threshold = threshold_otsu(gray)
    binary = gray > threshold
    return binary

def binarize_image(image_path):
    image = io.imread(image_path)
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = rgba2rgb(image)
        gray = rgb2gray(image)

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

    # Convert to uint8
    gray = (gray * 255).astype(np.uint8)

    # Calculate skew angle
    skew = calculate_skew(gray)
    print (f"Skew Angle: {skew}")

    # Correct skew
    deskewed = inter.rotate(gray, skew, reshape=False, order=0, mode='constant', cval=255/2)

    # Save the deskewed image
    deskewed_path = os.path.splitext(image_path)[0] + '_deskewed.png'
    deskewed = Image.fromarray(deskewed)
    deskewed.save(deskewed_path)
    return deskewed_path

def erase_left_margin(image_path, left_margin):
    image = cv2.imread(image_path)
    image[:, :left_margin] = 255 #
    return image

def detect_and_remove_boxes(image_path):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Use Canny edge detection
    edged = cv2.Canny(blurred, 50, 200, 255)

    # Perform dilation and erosion to close gaps in between object edges
    dilated_edged = cv2.dilate(edged.copy(), None, iterations=2)
    eroded_edged = cv2.erode(dilated_edged.copy(), None, iterations=1)

    # Find contours in the image
    contours, _ = cv2.findContours(eroded_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours
    for contour in contours:
        # Calculate aspect ratio, solidity, and extent of the contour
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        solidity = cv2.contourArea(contour) / float(w * h)
        extent = cv2.contourArea(contour) / float(cv2.contourArea(cv2.convexHull(contour)))

        # If the contour has an aspect ratio approximately equal to 1, and solidity and extent are high, it's likely a box
        if 0.9 <= aspect_ratio <= 1.1 and solidity > 0.9 and extent > 0.9:
            cv2.drawContours(image, [contour], -1, 255, 2)

    # Save the result
    result_path = os.path.splitext(image_path)[0] + '_boxes_removed.png'
    cv2.imwrite(result_path, image)

    return result_path

def remove_lines(image_path, left_margin, line_thickness=8):
    #remove the left margin
    image = erase_left_margin(image_path, left_margin)
    #convert the image to a numpy array
    image = np.array(image)
    #convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use Tesseract to get bounding boxes
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    n_boxes = len(d['level'])
    mask = np.zeros(image.shape, dtype=np.uint8)
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(mask, (x, y), (x + w, y + h), (255), -1)
    
    # Convert the mask to grayscale
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Now the mask should be in the correct format for cv2.inpaint
    image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA) # Use the Telea algorithm to inpaint the image
    #now detect lines and erase except for the mask area
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), line_thickness)

    #save the result to a file in the same directory as the image
    result = Image.fromarray(image)
    result_path = os.path.splitext(image_path)[0] + '_lines_removed.png'
    result.save(result_path)
    return result_path

    


    
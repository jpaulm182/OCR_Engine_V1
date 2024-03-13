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

def threshold_image(image_path, threshold=0.5) -> str:
    #threshold the image using the most robust and effective method
    image = io.imread(image_path)
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = rgba2rgb(image)
    gray = rgb2gray(image)
    thresh = threshold_otsu(gray)
    binary = gray > thresh
    binary = util.invert(binary)
    binary = util.img_as_ubyte(binary)
    binary = Image.fromarray(binary)
    #invert the image
    binary = ImageOps.invert(binary)
    binary_path = os.path.splitext(image_path)[0] + '_thresholded.png'
    binary.save(binary_path)
    return binary_path

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

    #save the grayscale image
    gray_path = os.path.splitext(image_path)[0] + '_gray.png'
    gray = Image.fromarray(gray)
    gray.save(gray_path)
    
    # Convert to uint8
    gray = np.array(Image.open(image_path))
    gray = (gray * 255).astype(np.uint8)

    # Calculate skew angle
    skew = calculate_skew(gray)
    print (f"Skew Angle: {skew}")

    # Correct skew
    deskewed = inter.rotate(gray, skew, reshape=True, order=1, mode='reflect')

    # Convert the float array to 8-bit
    deskewed = (deskewed * 255).astype(np.uint8)

    # Save the deskewed image
    deskewed_path = os.path.splitext(image_path)[0] + '_deskewed.png'
    deskewed = Image.fromarray(deskewed)
    deskewed.save(deskewed_path)    
    return deskewed_path

def erase_left_margin(image_path, left_margin):
    image = cv2.imread(image_path)
    image[:, :left_margin] = 255 #
    return image

def detect_and_visualize_boxes(image_path, line_thickness=2):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Create a copy of the image to draw on
    image_copy = image.copy()

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
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), 255, line_thickness)

    # Save the result
    result_path = os.path.splitext(image_path)[0] + '_detected_boxes.png'
    cv2.imwrite(result_path, image_copy)

    return result_path

def detect_and_remove_boxes(image_path, line_thickness=2, size_threshold=100, aspect_ratio_threshold=1.5):
    # Read the image in color
    image = cv2.imread(image_path)

    # Convert the image to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection
    edged = cv2.Canny(blurred, 50, 200, 255)

    # Perform dilation and erosion to close gaps in between object edges
    dilated_edged = cv2.dilate(edged.copy(), None, iterations=2)
    eroded_edged = cv2.erode(dilated_edged.copy(), None, iterations=1)

    # Find contours in the image
    contours, _ = cv2.findContours(eroded_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours
    for contour in contours:
        # Calculate bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Check if the contour is large enough to be a box and has a high enough aspect ratio
        if w * h > size_threshold and max(w, h) / min(w, h) > aspect_ratio_threshold:
            # Draw the rectangle on the image
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), line_thickness)

    # Save the result
    result_path = os.path.splitext(image_path)[0] + '_boxes_detected.png'
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

    


    
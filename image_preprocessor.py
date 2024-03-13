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

def detect_and_visualize_boxes(image_path, line_thickness=8, min_area=8000, min_aspect_ratio=.5, max_aspect_ratio=10):
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

        # Calculate the aspect ratio
        aspect_ratio = float(w) / h

        # Check if the area of the rectangle is above the minimum area and the aspect ratio is within the specified range
        if w * h > min_area and min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
            # Draw the rectangle on the image
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), line_thickness)

    # Save the result
    result_path = os.path.splitext(image_path)[0] + '_boxes_detected.png'
    cv2.imwrite(result_path, image)

    return result_path

import numpy as np

def detect_and_remove_boxes(image_path, line_thickness, min_area, min_aspect_ratio, max_aspect_ratio, inpaint_radius):
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

    # Create a mask for inpainting
    mask = np.zeros_like(gray)

    # Loop over the contours
    for contour in contours:
        # Calculate bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate the aspect ratio
        aspect_ratio = float(w) / h

        # Check if the area of the rectangle is above the minimum area and the aspect ratio is within the specified range
        if w * h > min_area and min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
            # Add the rectangle to the mask, but only the outline with the specified line thickness
            cv2.rectangle(mask, pt1 =(x, y), pt2=(x+w, y+h), color=255, thickness=line_thickness, lineType= 8)  #lineType is the thickness of the line so that the mask is not filled

    # Inpaint the image using the mask
    inpainted_image = cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)

    # Save the result
    result_path = os.path.splitext(image_path)[0] + '_boxes_removed.png'
    cv2.imwrite(result_path, inpainted_image)

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
    
    max_width = 500  # Maximum width of a bounding box
    max_height = 500  # Maximum height of a bounding box

    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        # Check if the bounding box is within the size limits
        if w <= max_width and h <= max_height:
            #color = white
            color = (255, 255, 255)
            cv2.rectangle(mask, pt1=(x, y), pt2=(x + w, y + h), color = color, thickness=-1, lineType=8)
    # Convert the mask to grayscale
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #save the mask as an image file
    #invert the mask
    mask = cv2.bitwise_not(mask)
    mask_path = os.path.splitext(image_path)[0] + '_line_mask.png'
    cv2.imwrite(mask_path, mask)
    
    #Start with the original image and change every pixel that is not in the mask to white
    image = gray
    #invert mask 
    mask = cv2.bitwise_not(mask)
    #analyze gray and turn all pixels that are not in the mask to white
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if mask[i][j] != 255:
                image[i][j] = 255
    #save the result to a file in the same directory as the image
    result = Image.fromarray(image)
    result_path = os.path.splitext(image_path)[0] + '_lines_removed.png'
    result.save(result_path)
       
    return result_path
    


    
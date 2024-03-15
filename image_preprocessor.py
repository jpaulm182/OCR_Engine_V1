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
    """
    Calculates the skew angle of an image.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    float: The skew angle of the image.
    """
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
    """
    Thresholds the input image using the most robust and effective method.

    Args:
        image_path (str): The path to the input image file.
        threshold (float): The threshold value for binarization. Defaults to 0.5.

    Returns:
        str: The path to the thresholded image file.
    """
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
    """
    Deskews an image by calculating the skew angle and correcting it.

    Args:
        image_path (str): The path to the input image file.

    Returns:
        str: The path to the deskewed image file.

    Raises:
        None

    """
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

def remove_lines(image_path, left_margin, min_width, min_height, max_width, max_height):
    """
    Removes lines from an image based on specified criteria.

    Args:
        image_path (str): The path to the input image file.
        left_margin (int): The width of the left margin to be removed.
        min_width (int): The minimum width of a bounding box to be considered.
        min_height (int): The minimum height of a bounding box to be considered.
        max_width (int): The maximum width of a bounding box to be considered.
        max_height (int): The maximum height of a bounding box to be considered.

    Returns:
        str: The path to the output image file with lines removed.
    """
    #remove the left margin
    image = erase_left_margin(image_path, left_margin)
    #convert the image to a numpy array
    image = np.array(image)
    #convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use Tesseract to get bounding boxes
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, lang='eng', config='--psm 6')
    n_boxes = len(d['level'])
    mask = np.zeros(image.shape, dtype=np.uint8)
    
    rectangles = []
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        # Check if the bounding box is within the size limits
        if min_width <= w <= max_width and min_height <= h <= max_height:
            rectangles.append([x, y, x + w, y + h])
    
    # Merge overlapping boxes
    #rectangles, weights = cv2.groupRectangles(rectangles, groupThreshold=1, eps=0.2)
    
    for rect in rectangles:
        #color = white
        color: tuple[Literal[255], Literal[255], Literal[255]] = (255, 255, 255)
        cv2.rectangle(mask, pt1=(rect[0], rect[1]), pt2=(rect[2], rect[3]), color = color, thickness=-1, lineType=8)
    
    # Convert the mask to grayscale
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #invert the mask
    mask = cv2.bitwise_not(mask)
    mask_path = os.path.splitext(image_path)[0] + '_line_mask.png'
    cv2.imwrite(mask_path, mask)
    
    #Start with the original image and change every pixel that is not in the mask to white
    image = gray
    #invert mask 
    mask = cv2.bitwise_not(mask)
    #analyze gray and turn all pixels that are not in the mask to white
    image[mask != 255] = 255
    #save the result to a file in the same directory as the image
    result = Image.fromarray(image)
    result_path = os.path.splitext(image_path)[0] + '_lines_removed.png'
    result.save(result_path)
       
    return result_path

    
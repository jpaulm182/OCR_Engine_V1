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
    thresh = threshold_otsu(noise_reduced) 
    binary = noise_reduced > thresh 

    # Convert to uint8
    binary = (binary * 255).astype(np.uint8)

    # Calculate the skew angle
    coords = np.column_stack(np.where(binary > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = -angle

    print(f"Skew adjustment: {angle} degrees")

    # Rotate the image to deskew it
    (h, w) = binary.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

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
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    contours = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = grab_contours(contours)
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
    for cnt in large_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 4)
    result = Image.fromarray(image)
    result_path = os.path.splitext(image_path)[0] + '_boxes_removed.png'
    result.save(result_path)
    return result_path

def remove_lines(image_path, left_margin, threshold, minLineLength, maxLineGap):
    #remove the left margin
    image = erase_left_margin(image_path, left_margin)
    #convert the image to a numpy array
    image = np.array(image)
    #Create a mask based on the regions with text so that the text is not modified
    mask = np.zeros(image.shape, dtype=np.uint8)
    #convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    #apply Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength, maxLineGap) #rho is the distance resolution of the accumulator in pixels, theta is the angle resolution of the accumulator in radians, threshold is the minimum number of intersections in a grid for a line to be detected, minLineLength is the minimum number of points that can form a line, maxLineGap is the maximum gap between two points to be considered in the same line
    # Assuming that 'image' is your input image
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    # Now you can draw lines on the mask
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(mask, (x1, y1), (x2, y2), (255), 8)
# Now the mask should be in the correct format for cv2.inpaint
    image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    #save the result to a file in the same directory as the image
    result = Image.fromarray(image)
    result_path = os.path.splitext(image_path)[0] + '_lines_removed.png'
    result.save(result_path)
    return result_path

    


    
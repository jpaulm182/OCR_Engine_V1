import pytesseract
from PIL import Image
import os
import cv2


def ocr_image(image_path: Image) -> str:
    
    image = Image.open(image_path)

    try:
        ocr_text_output= pytesseract.image_to_string(image, lang='eng', config= '--psm 4') #possible parameters: lang='eng', config='--psm 6'
        #config options are: --psm 6, --psm 11, --psm 12, --psm 13, --psm 4, --psm 7, --psm 8, --psm 9, --psm 10
        #the modes are: 6 = Assume a single uniform block of text, 11 = Sparse text, 12 = Sparse text with OSD, 13 = Raw line, 4 = Assume a single column of text of variable sizes, 7 = Treat the image as a single text line, 8 = Treat the image as a single word, 9 = Treat the image as a single word in a circle, 10 = Treat the image as a single character
        # The best for an outline format ocr is --psm 6
        #if nothing is specified, the default is --psm 3, which is auto mode and it does
        #--psm 3 specificaly means that the engine will try to fully automatic page segmentation which is the default mode
        #the psm modes are: 0 = Orientation and script detection (OSD) only, 1 = Automatic page segmentation with OSD, 2 = Automatic page segmentation, but no OSD, 3 = Fully automatic page segmentation, but no OSD, 4 = Assume a single column of text of variable sizes, 5 = Assume a single uniform block of vertically aligned text, 6 = Assume a single uniform block of text, 7 = Treat the image as a single text line, 8 = Treat the image as a single word, 9 = Treat the image as a single word in a circle, 10 = Treat the image as a single character
        print("OCR Complete for image")
    except Exception as e:
        raise Exception(f"Failed to perform OCR: {str(e)}")
    return ocr_text_output

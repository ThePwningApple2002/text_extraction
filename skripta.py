import cv2 as cv
import pytesseract
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import gc


def extract_by_layout_analysis(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    custom_config = r'--oem 3 --psm 11'
    data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)
    
    tracking_number = None
    napomena_value = None
    napomena_found = False
    
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if text and "PX" in text and "RS" in text and len(text) > 10:
            tracking_number = text
            
        if text.lower() in ["napomena", "napomena:"]:
            napomena_found = True
        elif napomena_found and text and text.strip().isdigit():
            napomena_value = text
            napomena_found = False  
    
    return tracking_number, napomena_value


pdf_path = "adrese.pdf" 

pages = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=5)

for i, page in enumerate(pages):
    print(f"Processing page {i+1}...")

    image = np.array(page)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    
    height, width = image.shape[:2]
    half_height = height // 2

    top_half = image[:half_height, :]
    bottom_half = image[half_height:, :]

    for part, half in enumerate([top_half, bottom_half], start=1):
        print(f"  Processing part {part} of page {i+1}...")

        tracking_number2, napomena_text2 = extract_by_layout_analysis(half)
        print(f"Tracking Number: {tracking_number2}")
        print(f"Napomena: {napomena_text2}")


    del image, top_half, bottom_half
    gc.collect()

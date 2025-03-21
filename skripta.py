import cv2 as cv
import pytesseract
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import gc
import csv


def extract_tracking_number(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    custom_config = r'--oem 3 --psm 11'
    data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)
    
    tracking_number = None
    
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if text and "PX" in text and "RS" in text and len(text) > 10:
            tracking_number = text
    return tracking_number

def extract_napomena(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    custom_config = r'--oem 3 --psm 11'
    data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)
    
    napomena_value = None
    napomena_found = False
    
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        
        if text.lower() in ["napomena", "napomena:"]:
            napomena_found = True
        elif napomena_found and text and text.strip().isdigit():
            napomena_value = text
            napomena_found = False  
    
    return napomena_value
pdf_path = "adrese.pdf" 
output_csv = "tracking_numbers.csv"

pages = convert_from_path(pdf_path, dpi=150, first_page=1)

results = []

for i, page in enumerate(pages):
    image = np.array(page)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    
    height, width = image.shape[:2]
    half_height = height // 2

    top_half = image[50:half_height, :]
    bottom_half = image[half_height:-50, :]

    tracking_number_top = top_half[50:150, 500:720]
    napomena_top = top_half[680:730, 820:900]
    
    tracking_number_bot = bottom_half[50:150, 500:720]
    napomena_bot = bottom_half[680:730, 820:900]

    # top_half_rgb = cv.cvtColor(tracking_number_bot, cv.COLOR_BGR2RGB)
    # bottom_half_rgb = cv.cvtColor(tracking_number_top, cv.COLOR_BGR2RGB)

    # top_pil = Image.fromarray(top_half_rgb)
    # bottom_pil = Image.fromarray(bottom_half_rgb)

    # top_pil.show()
    # bottom_pil.show()

    
    tracking_number_top_result = extract_tracking_number(tracking_number_top)
    napomena_top_result = extract_napomena(napomena_top)
    
    print(f"Tracking Number: {tracking_number_top_result}")
    print(f"Napomena: {napomena_top_result}")
    results.append([tracking_number_top_result, napomena_top_result])
    
    tracking_number_bot_result = extract_tracking_number(tracking_number_bot)
    napomena_bot_result = extract_napomena(napomena_bot)
    
    print(f"Tracking Number: {tracking_number_bot_result}")
    print(f"Napomena: {napomena_bot_result}")
    results.append([tracking_number_bot_result, napomena_bot_result])

    del image, top_half, bottom_half
    gc.collect()

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Tracking Number", "Napomena"])
    writer.writerows(results)

print(f"Results saved to {output_csv}")

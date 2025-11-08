import cv2 
import pytesseract 
# Charger une image de manuscrit 
img = cv2.imread('C://Users/PC/Pictures/img.jpg') 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
_, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY) 
# OCR avec Tesseract 
text = pytesseract.image_to_string(thresh, lang='fra') 
print(text)
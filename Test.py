import cv2 
# Extraction de caractéristiques SIFT 
sift = cv2.SIFT_create() 
img = cv2.imread('C://Users/PC/Pictures/img.jpg', cv2.IMREAD_GRAYSCALE) 
kp, des = sift.detectAndCompute(img, None) 
print("Nombre de points clés détectés :", len(kp)) 
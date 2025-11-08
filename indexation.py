import cv2

#Extraction de caractéristiques SIFT
sift = cv2.SIFT_create()
image = cv2.imread('C://Users/PC/Pictures/img.jpg', cv2.IMREAD_GRAYSCALE)
kp, des = sift.detectAndCompute(image, None)

print("Nombre de points clés détectés :", len(kp))
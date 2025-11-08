# TD 2 : Segmentation et détection 
# Outils : Python, OpenCV, scikit-image 
# Exercices : 
# 1. Segmentation par seuillage et morphologie (ouverture/fermeture). 
import cv2
import numpy as np
img = cv2.imread('Moliere.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((5,5), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # Ouverture (enlève le bruit)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel) # Fermeture (remplit les trous)
cv2.imshow('Original', gray)
cv2.imshow('Seuillage', thresh)
cv2.imshow('Ouverture', opening)
cv2.imshow('Fermeture', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2. Détection d’objets simples dans une vidéo (tracking couleur). 
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Filtrer les petits objets
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'OBJET', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.imshow('Tracking Couleur', frame)
    cv2.imshow('Masque', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# 3. Extraction de caractéristiques avec cv2.SIFT_create() ou ORB. 
img1 = cv2.imread('img1.jpg', 0)
img2 = cv2.imread('img2.jpg', 0)
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
print(f"Nombre de keypoints image 1: {len(kp1)}")
print(f"Nombre de keypoints image 2: {len(kp2)}")
print(f"Nombre de matches: {len(matches)}")
cv2.imshow('Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 4. Reconnaissance d’un objet à partir d’un modèle image. 
img = cv2.imread('chat3.jpg', 0)
template = cv2.imread('template1.jpg', 0)
w, h = template.shape[::-1]
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8  # Seuil de confiance
locations = np.where(res >= threshold)
img_color = cv2.imread('chat3.jpg')
for pt in zip(*locations[::-1]):
    cv2.rectangle(img_color, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
    cv2.putText(img_color, 'OBJET TROUVE', (pt[0], pt[1]-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
print(f"Nombre d'objets detectes: {len(locations[0])}")
cv2.imshow('Detection', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

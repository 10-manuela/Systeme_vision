# TD 1 : Manipulation d’images 
# Outils : Python, OpenCV, Matplotlib 
# Exercices : 
# 1. Charger et afficher une image (lecture avec cv2.imread et plt.imshow).
import cv2
img=cv2.imread("Moliere.jpg")
cv2.imshow('image originale', img)
cv2.waitKey(0)
cv2.destroyAllWindows

# 2. Convertir une image couleur en niveaux de gris. 
grayscale=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('image en niveau de gris', grayscale)
cv2.waitKey(0)
cv2.destroyAllWindows

# 3. Modifier la luminosité et le contraste. 
contraste=2
luminosite=0
img_modifier= cv2.convertScaleAbs(img, alpha=contraste, beta=luminosite)
cv2.imshow('image modifié (lumiosité et constraste)', img_modifier)
cv2.waitKey(0)
cv2.destroyAllWindows

# 4. Appliquer un flou moyen et un flou gaussien. 
gaussian = cv2.GaussianBlur(img, (5,5), 0)
cv2.imshow('image avec le flou Gaussian',gaussian)
cv2.waitKey(0)
cv2.destroyAllWindows

# 5. Détecter les contours avec les filtres Sobel et Canny. 
#Contours Sobel
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize =3)
sobely = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize =3)
sobel = cv2.magnitude(sobelx, sobely)
#Contours Canny
edges = cv2.Canny(img, 100, 200)
cv2.imshow('image avec  Détection de contour du filtre Sobel', sobel)
cv2.imshow('image avec  Détection de contours du filtre Canny', edges)
cv2.waitKey(0)
cv2.destroyAllWindows

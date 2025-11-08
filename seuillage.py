import matplotlib.pyplot as plt
import cv2

image = cv2.imread('C://Users/PC/Pictures/img.jpg', cv2.IMREAD_GRAYSCALE)

#seuillage simple
_,thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

#seuillage automatique
_,thresh2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.subplot(121), plt.imshow(thresh1, cmap='gray'), plt.title('Seuil fixe = 127')
plt.subplot(122), plt.imshow(thresh2, cmap='gray'), plt.title('Seuil Otsu')
plt.show()
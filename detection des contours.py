import cv2
import matplotlib.pyplot as plt 

image = cv2.imread('C://Users/PC/Pictures/bbb.jpg', cv2.IMREAD_GRAYSCALE)

#Contours Sobel
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize =3)
sobely = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize =3)
sobel = cv2.magnitude(sobelx, sobely)

#Contours Canny
edges = cv2.Canny(image, 100, 200)

plt.subplot(121), plt.imshow(sobel, cmap='gray'), plt.title('Sobel')
plt.subplot(122), plt.imshow(edges, cmap='gray'), plt.title('Canny')
plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('C://Users/PC/Pictures/aaa.jpg', cv2.IMREAD_GRAYSCALE)
#flou moyen
blur = cv2.blur(image, (5,5))

#flou gaussien 
gaussian = cv2.GaussianBlur(image, (5,5), 0)

#netteté
kernel_sharp = np.array([[0,-1,0],
                        [-1, 5, -1],
                        [0,-1,0]])
sharpen = cv2.filter2D(image, -1, kernel_sharp)
plt.figure(figsize=(10,4))
plt.subplot(131), plt.imshow(blur, cmap='gray'), plt.title('Flou moyen')
plt.subplot(132), plt.imshow(gaussian, cmap='gray'), plt.title('flou gaussian')
plt.subplot(133), plt.imshow(sharpen, cmap='gray'), plt.title('Netteté')
plt.show()
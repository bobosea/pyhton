import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('tablero.jpg',0)
img = np.float32(img)
dst = cv2.cornerHarris(img,2,3,0.04)
dst = cv2.dilate(dst,None)
plt.subplot(2,1,1), plt.imshow(dst )
plt.title('Harris Corner Detection'), plt.xticks([]), plt.yticks([])
plt.subplot(2,1,2),plt.imshow(img,cmap = 'gray')
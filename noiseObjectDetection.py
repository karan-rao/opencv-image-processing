import numpy as np
import cv2
import random

# Load the required image
img = cv2.imread("noiseImage.png",1)
cv2.imshow("Original",img)

# Convert BGR image to gray scale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Blur the image to reduce the noise
blur = cv2.GaussianBlur(gray,(5,5),0)
cv2.imshow("Blur",blur)

# Use adaptive threshold to segment the image
binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 325, 1)
cv2.imshow("Binary",binary)

# Find the contours on the binary image
contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filtering out the contours less than 1000 pixel^2
filtered = []
for c in contours:
	if cv2.contourArea(c) < 1000:continue
	filtered.append(c)

# Creating final output image
output = np.zeros([img.shape[0],img.shape[1],3], 'uint8')
for c in filtered:
	col = (150, 250, 250)
	cv2.drawContours(output,[c], -1, col, -1)
	area = cv2.contourArea(c)
	p = cv2.arcLength(c,True)
	print("Area: {} & Perimeter: {}".format(area,p))

cv2.imshow("Contours", output)

cv2.waitKey(0)
cv2.destroyAllWindows()
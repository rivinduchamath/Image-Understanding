import cv2
import numpy as np

# Load the image
image = cv2.imread("C:/Users/wogza/Desktop/exam/w.png", 1)
result11 = image.copy()
# convert BGR Image to HSV
hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# define range of color RED HSV
lower_red = np.array([161, 155, 84])
upper_red = np.array([179, 255, 255])

# Threshold the HSV image to get only red colors
mask = cv2.inRange(hsvImage, lower_red, upper_red)

# Bitwise-AND mask and original image
result = cv2.bitwise_and(image, image, mask=mask)
result2 = result.copy()

# Detect  contours on  image
contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
# draw all contours
drawing = np.zeros((hsvImage.shape[0], hsvImage.shape[1], 3), dtype=np.uint8)
CountersImg = cv2.drawContours(drawing, contours, -1, (0, 255, 0), 1)

# Find the index of the largest contour
areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cnt = contours[max_index]
x, y, w, h = cv2.boundingRect(cnt)
cv2.rectangle(CountersImg, (x, y), (x + w, y + h), (0, 255, 0), 2)

number_ofcontuers = len(contours)

cv2.imshow('All Red', result2)
cv2.imshow('Original', result11)
cv2.imshow('Contours', CountersImg)

print("Contour count ", number_ofcontuers)

cv2.waitKey()
cv2.destroyAllWindows()

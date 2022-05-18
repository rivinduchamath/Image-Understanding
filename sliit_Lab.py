import cv2
import numpy as np
import matplotlib.pyplot as plt

# Get Color Image
# Get Black and white image as 0
# Get unchanged the row version as -1
img = cv2.imread("C:/Users/wogza/Pictures/aa.jpg", 1)
cv2.imshow("Input Window", img)
cv2.waitKey()
cv2.destroyAllWindows()

# Change Colour Space
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(flags)

# convert to gray
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image", imgGray)

# convert to HSV
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV Image", imgGray)

cv2.waitKey()
cv2.destroyAllWindows()

# *************************************************************
# Lab 2
# *************************************************************

# Load or read the video
vid = cv2.VideoCapture("C:/Users/wogza/Downloads/qq.mp4")

# Take each Frame of the video
while vid.isOpened():
    ret, frame = vid.read()

    # Converting BGR to HSV color-space
    hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower rang and upper range of blue
    lower_Blue = np.array([110, 50, 50])
    upper_Blue = np.array([130, 255, 255])

    # There should be HSV image for a range of blue colour
    blueMusk = cv2.inRange(hsvImg, lower_Blue, upper_Blue)

    # Extract the blue object alone
    result = cv2.bitwise_and(frame, frame, mask=blueMusk)

    # Display Original Video
    originalVideo = cv2.resize(frame, (400, 400))
    cv2.imshow("Original Video", originalVideo)

    # Display Binary Video
    binaryVideo = cv2.resize(blueMusk, (400, 400))
    cv2.imshow("Binary Video", binaryVideo)

    # Display masked / object Video
    maskedVideo = cv2.resize(result, (400, 400))
    cv2.imshow("Masked Video", maskedVideo)

    # Condition to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

# ******************************************
# LAB 3
# Calculate Histograms

img = cv2.imread("C:/Users/wogza/Pictures/aa.jpg", 0)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# Simple Histogram For above image
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.plot(hist)
plt.show()

# Simple Histogram For above image OPENCV
img = cv2.imread("C:/Users/wogza/Pictures/aa.jpg", 0)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.plot(hist)
plt.show()

# Histogram calculation/ Plotting in NUMPY
img = cv2.imread("C:/Users/wogza/Pictures/aa.jpg", 0)
histr, bins = np.histogram(img.ravel(), 256, [0, 256])

# Show graph
plt.plot(histr)
plt.show()

# BGR histograms OPEN CV
img = cv2.imread("C:/Users/wogza/Pictures/aa.jpg", 1)
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 255])
plt.show()

# BGR histograms Numpy
img = cv2.imread("C:/Users/wogza/Pictures/aa.jpg", 1)
b, g, r = cv2.split(img)
histr1, bins = np.histogram(b.ravel(), 256, [0, 256])
histr2, bins = np.histogram(g.ravel(), 256, [0, 256])
histr3, bins = np.histogram(r.ravel(), 256, [0, 256])

plt.plot(histr1)
plt.plot(histr2)
plt.plot(histr3)
plt.show()

# BGR histograms Matplotlib
img = cv2.imread("C:/Users/wogza/Pictures/aa.jpg", 1)
b, g, r = cv2.split(img)

plt.hist(b.ravel(), 256, [0, 256])
plt.hist(g.ravel(), 256, [0, 256])
plt.hist(r.ravel(), 256, [0, 256])

plt.show()

# application of mask
img = cv2.imread("C:/Users/wogza/Pictures/aa.jpg", 0)
plt.imshow(img, "gray")
plt.show()

# Create a mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[50:400, 250:600] = 255

mask_img = cv2.bitwise_and(img, img, mask=mask)

# Calculate the histogram with mask and without mask
hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_mask = cv2.calcHist([img], [0], None, [256], [0, 256])

# Display Image
plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.subplot(223), plt.imshow(mask_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_full)

plt.xlim([0, 255])
plt.show()

# LAB 4 ++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Histogram Equalization to improve the contrast of a dark image

img = cv2.imread("C:/Users/wogza/Pictures/aa.jpg", 0)
histogram1 = cv2.calcHist([img], [0], None, [256], [0, 256])

# Creating a Histogram Equalization of a image using cv2.Equalization
equalizationImg = cv2.equalizeHist(img)
histogram2 = cv2.calcHist([equalizationImg], [0], None, [256], [0, 256])

# Complete the code here to plot the two histograms, dark and contrast
img = cv2.imread("C:/Users/wogza/Pictures/aa.jpg", 0)
histogram1 = cv2.calcHist([img], [0], None, [256], [0, 256])

# Creating histogram equalization of a image using cv2.equalization
equalizationImg = cv2.equalizeHist(img)
histogram2 = cv2.calcHist([img], [0], None, [256], [0, 256])

# complete the code here to plot the two histograms, dark and contrast adjust
plt.subplot(121), plt.plot(histogram1)
plt.subplot(121), plt.plot(histogram2)

result = np.hstack((img, equalizationImg))
cv2.imshow("Result", result)
cv2.waitKey(0)

# Histogram equalization to improve the contrast of a bright image
img = cv2.imread("C:/Users/wogza/Pictures/aa.jpg", 0)
histogram1 = cv2.calcHist([img], [0], None, [256], [0, 256])

# Creating a histogram equalization of a image using cv2.equalization()
equalizationImg = cv2.equalizeHist(img)
histogram2 = cv2.calcHist([equalizationImg], [0], None, [256], [0, 256])

# complete the code here to plot the two histograms, dark and contrast adjust
plt.subplot(211), plt.plot(histogram1)
plt.subplot(212), plt.plot(histogram2)

result = np.hstack((img, equalizationImg))
cv2.imshow("Result", result)
cv2.waitKey(0)

# Histogram equalization to improve the contrast of a dark and bright image
# Read images
# Dark Image
img1 = cv2.imread("C:/Users/wogza/Pictures/aa.jpg", 0)
# Bright Image
img2 = cv2.imread("C:/Users/wogza/Pictures/aa.jpg", 0)

histogram1 = cv2.calcHist([img], [0], None, [256], [0, 256])
histogram2 = cv2.calcHist([img], [0], None, [256], [0, 256])

# Creating a histogram equalization of a image using cv2.equalization()
equalizationImg1 = cv2.equalizeHist(img1)
equalizationImg2 = cv2.equalizeHist(img2)

histogram3 = cv2.calcHist([equalizationImg1], [0], None, [256], [0, 256])
histogram4 = cv2.calcHist([equalizationImg2], [0], None, [256], [0, 256])

# Complete the code here to plot the two histogram, dark and contrast adjust
plt.subplot(221), plt.plot(histogram1)
plt.subplot(222), plt.plot(histogram2)
plt.subplot(223), plt.plot(histogram3)
plt.subplot(224), plt.plot(histogram4)

plt.show()

# color histogram equalization to improve the contrast of color image
img = cv2.imread("C:/Users/wogza/Pictures/aa.jpg", 1)

# Convert image from RGB to HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Histogram equalization on the V-Channel
img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])

# convert image from HSV to RGB
img_RGB = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

# plot the histogram and display the two images
histogram3 = cv2.calcHist([img], [0], None, [256], [0, 256])
histogram4 = cv2.calcHist([img_RGB], [0], None, [256], [0, 256])

# complete the code here to plot the two histograms, dark and contrast adjust
plt.subplot(211), plt.plot(histogram1)
plt.subplot(212), plt.plot(histogram2)

result = np.hstack((img, img_RGB))
cv2.imshow('Result', result)
cv2.waitKey(0)

# LAB 5 ************************************************************
# Negative Transformation for color image
img = cv2.imread("C:/Users/wogza/Pictures/aa.jpg", 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_neg = cv2.bitwise_not(img)

# Display the image
plt.imshow(img)
plt.show()

plt.imshow(img_neg)  # Negative Image
plt.show()

# Power_Low transformation to improve the contrast of a dark image
img = cv2.imread('C:/Users/wogza/Pictures/aa.jpg', 0)
gamma = 0.6
img_gamma = np.power(img, gamma)

# Display Original image
plt.imshow(img, 'gray')
plt.show()

# Display Negative image
plt.imshow(img_gamma, 'gray')
plt.show()

# Log transformation to improve the dynamic range of an image
img = cv2.imread('C:/Users/wogza/Pictures/aa.jpg', 0)
c = 255 / np.log(1 + np.max(img))
np.seterr(divide='ignore')
log_img = c * (np.log(img + 1))

# specify the data type so that
log_image = np.array(log_img, dtype=np.uint8)

# Display both image
plt.imshow(img, 'gray')
plt.show()

# Log transformed image
plt.imshow(log_img, 'gray')
plt.show()

# Lab 6 *****************************************************

# Image filtering using 2D convolution
img = cv2.imread("C:/Users/wogza/Pictures/aa.jpg", 1)
kernal = np.ones((5, 5), np.float32) / 25

dst = cv2.filter2D(img, -1, kernal)

result = np.hstack((img, dst))
cv2.imshow('result', result)
cv2.waitKey(0)

# Image averaging using box filter
img = cv2.imread("C:/Users/wogza/Pictures/aa.jpg")
blur = cv2.blur(img, (5, 5))

result = np.hstack((img, dst))
cv2.imshow('result', result)

cv2.waitKey(0)

# Median filtering and gaussian filtering
# apply various low pass filters to smooth images
img = cv2.imread("C:/Users/wogza/Pictures/aa.jpg")
median = cv2.medianBlur(img, 3)

result = np.hstack((img, dst))
cv2.imshow("result", result)

cv2.waitKey(0)

# apply various low pass filters to smooth images
img = cv2.imread("C:/Users/wogza/Pictures/aa.jpg")
blur = cv2.GaussianBlur(img, (11, 11), 0)

result = np.hstack((img, dst))
cv2.imshow("result", result)

cv2.waitKey(0)

# Application of image blurring

# Load the image

img = cv2.imread("C:/Users/wogza/Pictures/aa.jpg")

# create custom kernel of size 15*15 and apply to the input image
kernel = np.ones((15, 15), np.float32) / 25

dst = cv2.filter2D(img, -1, kernal)

# Apply threshold operator to highlight the largest object
res, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

result = np.hstack((img, dst, thresh))
cv2.imshow('result', result)

cv2.waitKey(0)

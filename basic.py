import cv2 as cv

img = cv.imread('Resources/Photos/park.jpg')
cv.imshow('Boston', img)

# # converting to grayscale
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)
#
# #blurring
blur = cv.GaussianBlur(img, (1,15), cv.BORDER_DEFAULT)
# cv.imshow('Blur', blur)

# edge cascade
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny', canny)
canny2 = cv.Canny(img, 125, 175)
cv.imshow('Canny2', canny2)

#dilating the image
dilated = cv.dilate(canny, (3,3), iterations=5)
cv.imshow('Dylatacja', dilated)

#eroding the image
eroded = cv.erode(canny, (3,3), iterations=5)
cv.imshow('Erozja', eroded)

cv.waitKey(0)
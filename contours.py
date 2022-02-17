import  cv2 as cv
import numpy as np

img = cv.imread('Resources/Photos/group 2.jpg')

cv.imshow('Cats', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# blur = cv.GaussianBlur(img, (5, 5), cv.BORDER_DEFAULT)
# cv.imshow('Blue', blur)
#
# canny = cv.Canny(blur, 125, 175)
# cv.imshow('Canny', canny)

# threshold - progowanie
ret, thresh = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
cv.imshow('thresh', thresh)

# contours, hierarchies = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
contours, hierarchies = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

blank = np.zeros(img.shape)

cv.drawContours(blank, contours, -1, (0,255,0), 10)
cv.imshow('Contours Drawn', blank)

print(len(contours))

cv.waitKey(0)
import face_extract
import cv2 as cv
import numpy as np
from funkcje import *

#show dilate and erode
masked = cv.cvtColor(face_extract.masked, cv.COLOR_BGR2GRAY)
(thresh, im_bw) = cv.threshold(masked, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

kernel_3 = np.ones((3, 3), 'uint8')
kernel_5 = np.ones((7, 7), 'uint8')
kernel_7 = np.ones((15, 15), 'uint8')

dilate_img_3 = cv.dilate(masked, kernel_3, iterations=2)
dilate_img_5 = cv.dilate(masked, kernel_5, iterations=2)
dilate_img_7 = cv.dilate(masked, kernel_7, iterations=2)
dilate_img_3_bin = cv.dilate(im_bw, kernel_3, iterations=2)
dilate_img_5_bin = cv.dilate(im_bw, kernel_5, iterations=2)
dilate_img_7_bin = cv.dilate(im_bw, kernel_7, iterations=2)

opening_3 = cv.morphologyEx(masked, cv.MORPH_OPEN, kernel_3)
opening_3 = cv.morphologyEx(opening_3, cv.MORPH_OPEN, kernel_3)
opening_5 = cv.morphologyEx(masked, cv.MORPH_OPEN, kernel_5)
opening_5 = cv.morphologyEx(opening_5, cv.MORPH_OPEN, kernel_5)
opening_7 = cv.morphologyEx(masked, cv.MORPH_OPEN, kernel_7)
opening_7 = cv.morphologyEx(opening_7, cv.MORPH_OPEN, kernel_7)
opening_3_bin = cv.morphologyEx(im_bw, cv.MORPH_OPEN, kernel_3)
opening_3_bin = cv.morphologyEx(opening_3_bin, cv.MORPH_OPEN, kernel_3)
opening_5_bin = cv.morphologyEx(im_bw, cv.MORPH_OPEN, kernel_5)
opening_5_bin = cv.morphologyEx(opening_5_bin, cv.MORPH_OPEN, kernel_5)
opening_7_bin = cv.morphologyEx(im_bw, cv.MORPH_OPEN, kernel_7)
opening_7_bin = cv.morphologyEx(opening_7_bin, cv.MORPH_OPEN, kernel_7)

closing_3 = cv.morphologyEx(masked, cv.MORPH_CLOSE, kernel_3)
# closing_3 = cv.morphologyEx(closing_3, cv.MORPH_CLOSE, kernel_3)
closing_5 = cv.morphologyEx(masked, cv.MORPH_CLOSE, kernel_5)
# closing_5 = cv.morphologyEx(closing_5, cv.MORPH_CLOSE, kernel_5)
closing_7 = cv.morphologyEx(masked, cv.MORPH_CLOSE, kernel_7)
# closing_7 = cv.morphologyEx(closing_7, cv.MORPH_CLOSE, kernel_7)
closing_3_bin = cv.morphologyEx(im_bw, cv.MORPH_CLOSE, kernel_3)
# closing_3_bin = cv.morphologyEx(closing_3_bin, cv.MORPH_CLOSE, kernel_3)
closing_5_bin = cv.morphologyEx(im_bw, cv.MORPH_CLOSE, kernel_5)
# closing_5_bin = cv.morphologyEx(closing_5_bin, cv.MORPH_CLOSE, kernel_5)
closing_7_bin = cv.morphologyEx(im_bw, cv.MORPH_CLOSE, kernel_7)
# closing_7_bin = cv.morphologyEx(closing_7_bin, cv.MORPH_CLOSE, kernel_7)

# cv.imshow('otwarcie 3', closing_3)
# cv.imshow('otwarcie 5', closing_5)
# cv.imshow('otwarcie 7', closing_7)
# cv.imshow('otwarcie 3 bin', closing_3_bin)
# cv.imshow('otwarcie 5 bin', closing_5_bin)
# cv.imshow('otwarcie 7 bin', closing_7_bin)
# cv.imshow('dylatacja 3 bin', dilate_img_3_bin)
# cv.imshow('dylatacja 5 bin', dilate_img_5_bin)
# cv.imshow('dylatacja 7 bin', dilate_img_7_bin)


img2 = cv.imread('Resources/Photos/gwiazda.jpg')
img3 = cv.imread('Resources/Photos/prostokat.jpg')
img4 = cv.imread('Resources/Photos/giwzda2.png')
gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)


cv.imshow('szkielet1', gray)
thinned = cv.ximgproc.thinning(gray, thinningType=cv.ximgproc.THINNING_GUOHALL)
cv.imshow('szkielet2', thinned)


gray = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)
cv.imshow('szkielet11', gray)
thinned = cv.ximgproc.thinning(gray, thinningType=cv.ximgproc.THINNING_GUOHALL)
cv.imshow('szkielet21', thinned)
cv.waitKey()
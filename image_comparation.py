import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageChops


img = cv.imread('base_of_sings_2/Szkielet32.jpg', 0)
img_compare = cv.imread('base_of_sings_2/Szkielet49.jpg', 0)

# metoda 1
# cv.waitKey(0)
# #--- take the absolute difference of the images ---
# res = cv.absdiff(img, img_compare)
#
# # #--- convert the result to integer type ---
# res = res.astype(np.uint8)
# # #--- find percentage difference based on number of pixels that are not zero ---
# percentage = (np.count_nonzero(res) * 100)/ res.size
#
# print("perc", percentage)

# metoda 2

# ret, thresh = cv.threshold(img, 127, 255,0)
# ret, thresh2 = cv.threshold(img_compare, 127, 255,0)
# contours,hierarchy = cv.findContours(thresh,2,1)
# cnt1 = contours[0]
# contours,hierarchy = cv.findContours(thresh2,2,1)
# cnt2 = contours[0]
# ret = cv.matchShapes(cnt1,cnt2,1,0.0)
# print( ret )

# metoda 3
# print(img.size, img_compare.size)
# s = ssim(img, img_compare)
# print(s)
#

img_1 = Image.open('base_of_sings_2/Szkielet_32.jpg')
img_2 = Image.open('base_of_sings_2/Szkielet_35.jpg')
different = ImageChops.difference(img_1, img_2)
different.show()
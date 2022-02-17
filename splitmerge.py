import  cv2 as cv
import numpy as np

img = cv.imread('Resources/Photos/park.jpg')
cv.imshow('Boston', img)

b, g, r = cv.split(img)

print(img.shape)
print(b.shape)
print(r.shape)
print(g.shape)

blank = np.zeros(img.shape[:2], dtype='uint8')
merged = cv.merge([b, r, g])

r_2 = np.zeros(b.shape, dtype='uint8')


blue = cv.merge([b, r_2, blank])
# green = cv.merge([blank, g, blank])
# red = cv.merge([blank,blank,r])
# #
# cv.imshow('Merged', merged)
cv.imshow('Merged', blue)
# cv.imshow('Merged', green)
# cv.imshow('Merged', red)


cv.waitKey(0)
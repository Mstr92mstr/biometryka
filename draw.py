import cv2 as cv
import numpy as np

blank = np.zeros((500,500,3), dtype='uint8')
# cv.imshow('Pusty', blank)

img = cv.imread('Resources/Photos/lady.jpg')
# cv.imshow('Lady', img)

blank[200:300, 300:400] = 0,255,0
cv.imshow('Green', blank)

cv.putText(blank, "Spierdalaj stara kurwo", (100,355), cv.FONT_HERSHEY_SCRIPT_COMPLEX, 1.0, (255, 255, 255), 5)
cv.imshow('Text', blank)

cv.waitKey(0)
import cv2 as cv
import numpy as np
from funkcje import usuniecie_tla

kaskada = 'haar_face.xml'
img = cv.imread('Resources/Photos/RAINBOW/37.jpg')
blank = np.zeros(img.shape[:2], dtype='uint8')
cv.imshow('zero', blank)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('etap 1 transformacja do odcieni szarosci', gray)
haar_cascade = cv.CascadeClassifier(kaskada)
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=3)
#generacja pustego zdjecia o wymiarach zdjecia wejsciowego
blank = np.zeros(img.shape[:2], dtype='uint8')
# faces rect to lista zmiennych gdzie kolejno x, y, w, h
if len(faces_rect) == 1:
    for (x, y, w, h) in faces_rect:
        radius = int(w*0.66)
        cent_x = int(x+w/2)
        cent_y = int(y+w/2)
        print('center', x+radius, cent_y)
        marked_face = cv.circle(img, (cent_x, cent_y), radius, (0, 0, 255), 1)
        cv.imshow('etap 2 wykrycie twarzy kaskada haara', marked_face)
else:
    print('Nie wykryto lub wykryto więcej niż jedną twarz, zmień zdjęcie!')

mask = cv.circle(blank, (cent_x, cent_y), radius, 255, -1)
#usuniecie zimnego tla
bacground = usuniecie_tla(img)
cv.imshow('etap 3 usuniecie zimnego tla', bacground)

masked = cv.bitwise_and(bacground, bacground, mask=mask)
cv.imshow('etap 4 nalozenie maski', masked)

# blur = cv.GaussianBlur(masked, (5,5), cv.BORDER_DEFAULT)
# cv.imshow('etap 5 rozmycie Gaussowskie', blur)

canny = cv.Canny(masked, 125, 175)
cv.imshow('etap 5 Canny krawedzie', canny)

cv.waitKey(0)

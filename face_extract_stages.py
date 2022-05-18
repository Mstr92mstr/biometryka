import cv2 as cv
import numpy as np
from funkcje import *


img = cv.imread('Resources/Photos/RAINBOW/37.jpg')
#generacja pustego zdjecia o wymiarach zdjecia wejsciowego
blank = np.zeros(img.shape[:2], dtype='uint8')
kaskada = 'haar_face.xml'
#zmiana na odcienie szarosci do wykrycia twarzy przez kaskade
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('etap 1 transformacja do odcieni szarosci', gray)
haar_cascade = cv.CascadeClassifier(kaskada)
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=3)

# wykonanie extrakcji twarzy pod warunkiem wykrycia 1 twarzy
if len(faces_rect) == 1:
    #wyznaczenie środka okręgu wykrytej twarzy
    mask, marked_face = twarz_maska(img, faces_rect)
    cv.imshow('etap 2 wykrycie twarzy kaskada haara', marked_face)

    #usuniecie zimnego tla
    background = usuniecie_tla(img)
    cv.imshow('etap 3 usuniecie zimnego tla', background)

    masked = cv.bitwise_and(background, background, mask=mask)
    cv.imshow('etap 4 nalozenie maski', masked)
    #zapis wyextractowanej twarzy
    cv.imwrite('extracted_face.jpg', masked)
    # blur = cv.GaussianBlur(masked, (5,5), cv.BORDER_DEFAULT)
    # cv.imshow('etap 5 rozmycie Gaussowskie', blur)

    canny = cv.Canny(masked, 125, 175)
    cv.imshow('etap 5 Canny krawedzie', canny)

else:
    print('Nie wykryto lub wykryto więcej niż jedną twarz, zmień zdjęcie!')

cv.waitKey(0)

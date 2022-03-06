import cv2 as cv
import numpy as np
from funkcje import *


img = cv.imread('Resources/Photos/RAINBOW/49.jpg')
cv.imshow('Zdjecie wejsciowe', img)
#generacja pustego zdjecia o wymiarach zdjecia wejsciowego
blank = np.zeros(img.shape[:2], dtype='uint8')
kaskada = 'haar_face.xml'
#zmiana na odcienie szarosci do wykrycia twarzy przez kaskade
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
haar_cascade = cv.CascadeClassifier(kaskada)
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=3)

# wykonanie extrakcji twarzy pod warunkiem wykrycia 1 twarzy
if len(faces_rect) == 1:
    #wyznaczenie środka okręgu wykrytej twarzy
    mask, marked_face = twarz_maska(img, faces_rect, blank)

    #usuniecie zimnego tla
    background = usuniecie_tla(img)

    masked = cv.bitwise_and(background, background, mask=mask)
    cv.imshow('wydzielona twarz', masked)
    #zapis wyextractowanej twarzy
    cv.imwrite('extracted_face.jpg', masked)
elif len(faces_rect) == 0:
    print('Nie wykryto twarzy, zmień zdjęcie!')
else:
    print('Wykryto więcej niż jedną twarz, zmień zdjęcie!')

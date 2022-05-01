import cv2 as cv
import numpy as np
from funkcje import *


img = cv.imread('Resources/Photos/RAINBOW/37.jpg')
cv.imshow('Zdjecie wejsciowe', img)
#generacja pustego zdjecia o wymiarach zdjecia wejsciowego
blank = np.zeros(img.shape[:2], dtype='uint8')
kaskada = 'haar_face.xml'
# kaskada = 'haarcascade_frontalface_alt.xml'
#zmiana na odcienie szarosci do wykrycia twarzy przez kaskade
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
haar_cascade = cv.CascadeClassifier(kaskada)
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=3)

print("wykryto", len(faces_rect), "twarz/twarze")
# wykonanie extrakcji twarzy pod warunkiem wykrycia 1 twarzy
if len(faces_rect) == 2:
    # odjęcie falszywie wykrytej twarzy - najpierw utworzenie podział na pojedyncze wektory
    faces_rect1 = np.delete(faces_rect, 0, axis=0)
    faces_rect2 = np.delete(faces_rect, 1, axis=0)
    #wybor odrzuconego wykrycia
    faces_rect = np.delete(faces_rect, odrzucenie_wykrycia(faces_rect1, faces_rect2), axis=0)
    mask, marked_face = twarz_maska(img, faces_rect, blank)
    # usuniecie zimnego tla
    background = usuniecie_tla(img)
    masked = cv.bitwise_and(background, background, mask=mask)
    # zapis wyextractowanej twarzy
    cv.imwrite('extracted_face.jpg', masked)
elif len(faces_rect) == 1:
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

# print(faces_rect)
# faces_rect = np.delete(faces_rect, 0, axis=0)
# faces_rect = np.delete(faces_rect, 0, axis=0)
# print(faces_rect)
# mask, marked_face = twarz_maska(img, faces_rect, blank)
# # usuniecie zimnego tla
# background = usuniecie_tla(img)
# masked = cv.bitwise_and(background, background, mask=mask)
# # zapis wyextractowanej twarzy
# cv.imwrite('extracted_face.jpg', masked)

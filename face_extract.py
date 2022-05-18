import cv2 as cv
import numpy as np
from funkcje import *


img = cv.imread('Resources/Photos/RAINBOW/3.jpg')
cv.imshow('Zdjecie wejsciowe', img)
#generacja pustego zdjecia o wymiarach zdjecia wejsciowego
puste_zdjecie = np.zeros(img.shape[:2], dtype='uint8')
kaskada = 'haar_face.xml'
# kaskada = 'haarcascade_frontalface_alt.xml'
#zmiana na odcienie szarosci do wykrycia twarzy przez kaskade
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
haar_cascade = cv.CascadeClassifier(kaskada)
wykryte_twarze = haar_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4)
print("wykryto", len(wykryte_twarze), "twarz/twarze")
# wykonanie extrakcji twarzy pod warunkiem wykrycia 1 twarzy
if len(wykryte_twarze) == 2:
    # odjęcie falszywie wykrytej twarzy - najpierw utworzenie podział na pojedyncze wektory
    twarz_prostokat1 = np.delete(wykryte_twarze, 0, axis=0)
    twarz_prostokat2 = np.delete(wykryte_twarze, 1, axis=0)
    #wybor odrzuconego wykrycia
    wykryte_twarze = np.delete(wykryte_twarze, odrzucenie_wykrycia(twarz_prostokat1, twarz_prostokat2), axis=0)
    maska, zaznaczona_twarz = twarz_maska(img, wykryte_twarze)
    # usuniecie zimnego tla
    tlo = usuniecie_tla(img)
    po_maskowaniu = cv.bitwise_and(tlo, tlo, mask=maska)
    # zapis wyextractowanej twarzy
    cv.imwrite('extracted_face.jpg', po_maskowaniu)
elif len(wykryte_twarze) == 1:
    #wyznaczenie środka okręgu wykrytej twarzy
    maska, zaznaczona_twarz = twarz_maska(img, wykryte_twarze)
    #usuniecie zimnego tla
    tlo = usuniecie_tla(img)
    # cv.imshow('tlo', tlo)
    po_maskowaniu = cv.bitwise_and(tlo, tlo, mask=maska)
    # cv.imshow('wydzielona twarz', po_maskowaniu)
    #zapis wyextractowanej twarzy
    cv.imwrite('extracted_face.jpg', po_maskowaniu)
elif len(wykryte_twarze) == 0:
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

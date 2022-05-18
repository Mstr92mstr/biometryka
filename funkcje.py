import cv2 as cv
import numpy as np


def usuniecie_tla(img):
    # Zdefiniowanie dolnej i górnej granicy wycinanego koloru niebieskiego
    gorny = np.array([0, 0, 0])
    dolny = np.array([255, 150, 150])
    # stworzenie maski wycinajacej tlo
    zakres = cv.inRange(img, gorny, dolny)
    # aplikacja przekształceń morfologicznych
    jadro = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    morfologia = cv.morphologyEx(zakres, cv.MORPH_CLOSE, jadro)
    # odwrócenie wyników
    maska = 255 - morfologia
    # cv.imshow('maska', maska)
    # zastosowanie maski do obrazu wejściowego
    wynik = cv.bitwise_and(img, img, mask=maska)
    return wynik


def twarz_maska(img, faces_rect):
    puste_zdjecie = np.zeros(img.shape[:2], dtype='uint8')
    for (x, y, w, h) in faces_rect:
        radius = int(w*0.56)
        cent_x = int(x+w/2)
        cent_y = int(y+w/2)
        marked_face = cv.circle(img, (cent_x, cent_y), radius, (0, 0, 255), 1)
        #generacja maski
        mask = cv.circle(puste_zdjecie, (cent_x, cent_y), radius, 255, -1)
        return mask, marked_face

def odrzucenie_wykrycia(wykrycie1, wykrycie2):
    len_1 = wykrycie1[0, 2]
    len_2 = wykrycie2[0, 2]
    if len_1 > len_2:
        return 0
    else:
        return 1

def podzial_segmentacja(img, kolor):
    # Define lower and uppper limits
    lower = np.array(kolor - 1)
    upper = np.array([kolor + 1])
    # Create mask to only select color from input
    thresh = cv.inRange(img, lower, upper)
    # apply morphology
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 1))
    morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    result = cv.bitwise_and(img, img, mask=morph)
    return result
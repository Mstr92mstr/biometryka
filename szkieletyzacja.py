import cv2 as cv
import numpy as np
from funkcje import *


def generacja_sladu(img):
    cv.imshow('Zdjecie wejsciowe', img)
    # generacja pustego zdjecia o wymiarach zdjecia wejsciowego
    puste_zdjecie = np.zeros(img.shape[:2], dtype='uint8')
    kaskada = 'haar_face.xml'
    # kaskada = 'haarcascade_frontalface_alt.xml'
    # zmiana na odcienie szarosci do wykrycia twarzy przez kaskade
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    haar_cascade = cv.CascadeClassifier(kaskada)
    wykryte_twarze = haar_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2)
    # wykryte_twarze = haar_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4)
    print("wykryto", len(wykryte_twarze), "twarz/twarze")
    # wykonanie extrakcji twarzy pod warunkiem wykrycia 1 twarzy
    if len(wykryte_twarze) == 2:
        # odjęcie falszywie wykrytej twarzy - najpierw utworzenie podział na pojedyncze wektory
        twarz_prostokat1 = np.delete(wykryte_twarze, 0, axis=0)
        twarz_prostokat2 = np.delete(wykryte_twarze, 1, axis=0)
        # wybor odrzuconego wykrycia
        wykryte_twarze = np.delete(wykryte_twarze, odrzucenie_wykrycia(twarz_prostokat1, twarz_prostokat2), axis=0)
        maska, zaznaczona_twarz = twarz_maska(img, wykryte_twarze)
        # usuniecie zimnego tla
        tlo = usuniecie_tla(img)
        po_maskowaniu = cv.bitwise_and(tlo, tlo, mask=maska)
        # zapis wyextractowanej twarzy
        # cv.imwrite('extracted_face.jpg', po_maskowaniu)
    elif len(wykryte_twarze) == 1:
        # wyznaczenie środka okręgu wykrytej twarzy
        maska, zaznaczona_twarz = twarz_maska(img, wykryte_twarze)
        # usuniecie zimnego tla
        tlo = usuniecie_tla(img)
        po_maskowaniu = cv.bitwise_and(tlo, tlo, mask=maska)
        cv.imwrite('extracted_face.jpg', po_maskowaniu)
    elif len(wykryte_twarze) == 0:
        raise ValueError('Nie wykryto twarzy, zmień zdjęcie!')
    else:
        # print('exception!!!!')
        raise ValueError('Wykryto więcej niż jedną twarz, zmień zdjęcie!')

    img = po_maskowaniu
    twoDimage = img.reshape((-1, 3))
    twoDimage = np.float32(twoDimage)
    # cv.imshow('twoDimage', twoDimage)
    kryteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # liczba ekstraktowanych kolorow do segmentacji
    K = 4
    podejscia = 10
    segmentacja_maska = []

    ret, label, center = cv.kmeans(twoDimage, K, None, kryteria, podejscia, cv.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    image_segmentation = res.reshape(img.shape)

    # cv.imshow('Twarz po segmentacji', image_segmentation)
    n = 0
    for kolor in center:
        # print('kolor', kolor)
        segmentacja_maska.append(podzial_segmentacja(image_segmentation, kolor))

    for n in range(K):
        tekst = ('maska ') + str(n)
        # cv.imshow(tekst, segmentacja_maska[n])
        kernel_3 = np.ones((2, 2), 'uint8')
        closing_3 = cv.morphologyEx(segmentacja_maska[n], cv.MORPH_CLOSE, kernel_3)
        opening_3 = cv.morphologyEx(closing_3, cv.MORPH_OPEN, kernel_3)
        tekst_2 = tekst + 'po morfologi'
        # cv.imshow(tekst_2, opening_3)
    # wpisanie pustej maski do wyniku operacji szkieletonizacji
    result = cv.cvtColor(segmentacja_maska[0], cv.COLOR_BGR2GRAY)
    # cv.waitKey()
    # koza = [0, 2, 3]
    for n in range(1, K):
        # for n in koza:
        gray = cv.cvtColor(segmentacja_maska[n], cv.COLOR_BGR2GRAY)
        kernel_3 = np.ones((3, 3), 'uint8')
        closing_3 = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel_3)
        opening_3 = cv.morphologyEx(closing_3, cv.MORPH_OPEN, kernel_3)
        threshold, warstwa = cv.threshold(opening_3, 10, 255, cv.THRESH_BINARY)
        szkielet = cv.ximgproc.thinning(warstwa, thinningType=cv.ximgproc.THINNING_GUOHALL)
        result = cv.bitwise_or(result, szkielet)
    cv.imshow('Szkielet', result)
    cv.imwrite('test_systemu.jpg', result)
    cv.waitKey(0)
    return result
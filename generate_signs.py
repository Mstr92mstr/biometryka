import segmentation
import cv2 as cv
from funkcje import *
import numpy as np

for i in range(32, 58):
    nazwa_odczyt_zdjecie = 'Resources/Photos/RAINBOW/' + str(i) + '.jpg'
    # nazwa_odczyt_zdjecie = 'Resources/Photos/RAINBOW/' + '32' + '.jpg'
    img = cv.imread(nazwa_odczyt_zdjecie)
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


    img = masked

    twoDimage = img.reshape((-1,3))
    twoDimage = np.float32(twoDimage)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # liczba ekstraktowanych kolorow do segmentacji
    K = 4
    attempts = 10
    segmentacja_maska = []

    ret, label, center = cv.kmeans(twoDimage, K, None, criteria, attempts, cv.KMEANS_PP_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    image_segmentation = res.reshape(img.shape)

    cv.imshow('Twarz po segmentacji', image_segmentation)
    n = 0
    for kolor in center:
        print('kolor', kolor)
        segmentacja_maska.append(podzial_segmentacja(image_segmentation, kolor))

    for n in range(K):
        tekst = ('maska ') + str(n)
        cv.imshow(tekst, segmentacja_maska[n])

    #wpisanie pustej maski do wyniku operacji szkieletonizacji
    result = cv.cvtColor(segmentation.segmentacja_maska[0], cv.COLOR_BGR2GRAY)

    for n in range(1, segmentation.K):
        gray = cv.cvtColor(segmentacja_maska[n], cv.COLOR_BGR2GRAY)
        kernel_3 = np.ones((3, 3), 'uint8')
        closing_3 = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel_3)
        opening_3 = cv.morphologyEx(closing_3, cv.MORPH_OPEN, kernel_3)
        threshold, thresh = cv.threshold(opening_3, 10, 255, cv.THRESH_BINARY)
        thinned = cv.ximgproc.thinning(thresh, thinningType=cv.ximgproc.THINNING_GUOHALL)
        cv.imshow('warstwa', gray)
        cv.imshow('thinned', thinned)
        cv.waitKey()
        print('size', thinned.size, result.size, thresh.size, gray.size)
        result = cv.bitwise_or(result, thinned)

    cv.imshow('Wynik', result)
    nazwa_zapis = 'images_with_signs/signed_' + str(i) + '.jpg'
    # nazwa_zapis = 'signs_cleared/signed_' + 'test' + '.jpg'
    cv.imwrite(nazwa_zapis, result)
    cv.waitKey(0)




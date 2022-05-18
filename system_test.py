import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt
from szkieletyzacja import *

# porownanie
licznosc_bazy = 26
wartosci_porownania_x = []
x = []


def porownanie_1(img1, img2):
    res = cv.absdiff(img1, img2)
    res = res.astype(np.uint8)
    ret = 1 - np.count_nonzero(res) / res.size
    return ret


def porownanie_2(img1, img2):
    ret, thresh = cv.threshold(img1, 127, 255, 0)
    ret, thresh2 = cv.threshold(img2, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, 2, 1)
    cnt1 = contours[0]
    contours, hierarchy = cv.findContours(thresh2, 2, 1)
    cnt2 = contours[0]
    ret = cv.matchShapes(cnt1, cnt2, 1, 0.0)
    return ret


flag = False
flaga_exit = False

obraz_wejsciowy_raw = cv.imread('Resources/Photos/RAINBOW/34.jpg')
try:
    obraz_wejsciowy = generacja_sladu(obraz_wejsciowy_raw)
except Exception as e:
    print(e)
    flaga_exit = True

pusty = cv.imread('base_of_sings_2/Szkielet_pusty.jpg')
cv.waitKey()
if flaga_exit is False:
    for j in range(0, licznosc_bazy):
        nazwa_odczyt_baza = 'base_of_sings_2/Szkielet_' + str(j + 32) + '.jpg'
        img_baza = cv.imread(nazwa_odczyt_baza, 0)
        score, diff = ssim(obraz_wejsciowy, img_baza, full=True)
        x.append(j)
        metoda_3 = score
        wartosci_porownania_x.append(metoda_3)
    # print(wartosci_porownania_x)
    for i in wartosci_porownania_x:
        if i > 0.9:
            print("Przyznano dostęp")
            flag = True
            break
    if flag is not True:
        print("Brak użytkownika w bazie")

    plt.stem(x, wartosci_porownania_x)
    plt.autoscale()
    # plt.show()
else:
    print("Przerwano działanie algorytmu, zmień zdjęcie!")


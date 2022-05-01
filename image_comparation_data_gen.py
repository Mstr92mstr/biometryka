import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt


# porownanie
licznosc_bazy = 26
wartosci_porownania_x = []
x = []

def porownanie_1(img1, img2):
    res = cv.absdiff(img1, img2)
    res = res.astype(np.uint8)
    ret = 1 - np.count_nonzero(res)/ res.size
    return ret


def porownanie_2(img1, img2):
    ret, thresh = cv.threshold(img1, 127, 255,0)
    ret, thresh2 = cv.threshold(img2, 127, 255,0)
    contours,hierarchy = cv.findContours(thresh,2,1)
    cnt1 = contours[0]
    contours,hierarchy = cv.findContours(thresh2,2,1)
    cnt2 = contours[0]
    ret = cv.matchShapes(cnt1,cnt2,1,0.0)
    return ret


for i in range(0, licznosc_bazy):
    nazwa_odczyt_porownywany = 'base_of_sings/Szkielet' + str(i+32) + '.jpg'
    img_podpis = cv.imread(nazwa_odczyt_porownywany, 0)
    for j in range(0, licznosc_bazy):
        nazwa_odczyt_baza = 'base_of_sings/Szkielet' + str(j+32) + '.jpg'
        img_baza = cv.imread(nazwa_odczyt_baza, 0)
        # metoda 1 calculate absolute difference of images (by pixels)
        metoda_1 = porownanie_1(img_podpis, img_baza)
        # metoda 2 findContours tragedia
        metoda_2 = porownanie_2(img_podpis, img_baza)
        # metoda 3 The Structural Similarity Index
        metoda_3 = ssim(img_podpis, img_baza)
        x.append(j)
        wartosci_porownania_x.append(metoda_1)
    # print(wartosci_porownania_x)
    plt.stem(x, wartosci_porownania_x)
    plt.autoscale()
    plt.show()
    # np.r_(wartosci_porownania_y, wartosci_porownania_x)
    # wartosci_porownania_y = np.stack([wartosci_porownania_y, wartosci_porownania_x])
    # wartosci_porownania_y = np.append(wartosci_porownania_y, wartosci_porownania_x, axis=0)
    # np.append(wartosci_porownania_y, wartosci_porownania_x, axis=i)
    # np.append(wartosci_porownania_y, wartosci_porownania_x, axis=0)
    # wartosci_porownania_x.clear()
    # np.insert(wartosci_porownania_y, i, wartosci_porownania_x)
    wartosci_porownania_x.clear()
    x.clear()
# print(wartosci_porownania_y)
print(type(wartosci_porownania_x))
# print(type(wartosci_porownania_y))


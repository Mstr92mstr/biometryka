import cv2 as cv


for i in range(32, 58):
    nazwa_odczyt_zdjecie = 'Resources/Photos/RAINBOW/' + str(i) + '.jpg'
    nazwa_odczyt_podpis = 'base_of_sings/Szkielet' + str(i) + '.jpg'
    nazwa_zapis = 'images_with_signs/signed_' + str(i) + '.jpg'

    img_zdj = cv.imread(nazwa_odczyt_zdjecie)
    img_podpis = cv.imread(nazwa_odczyt_podpis)

    print(nazwa_odczyt_zdjecie, nazwa_odczyt_podpis)
    cv.waitKey(0)

    cv.imshow('zdjecie', img_zdj)
    cv.imshow('podpis', img_podpis)
    cv.waitKey(0)
    #
    print(img_zdj.size, img_podpis.size)
    nalozenie = cv.bitwise_or(img_zdj, img_podpis)
    cv.imwrite(nazwa_zapis, nalozenie)


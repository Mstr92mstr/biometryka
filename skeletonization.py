import segmentation
import cv2 as cv
import numpy as np


def generacja_sladu(obraz_wejsciowy):

    # wpisanie pustej maski do wyniku operacji szkieletonizacji
    result = cv.cvtColor(segmentation.segmentacja_maska[0], cv.COLOR_BGR2GRAY)
    # koza = [0, 2, 3]
    for n in range(1, segmentation.K):
    # for n in koza:
        gray = cv.cvtColor(segmentation.segmentacja_maska[n], cv.COLOR_BGR2GRAY)
        kernel_3 = np.ones((3, 3), 'uint8')
        closing_3 = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel_3)
        opening_3 = cv.morphologyEx(closing_3, cv.MORPH_OPEN, kernel_3)
        threshold, warstwa = cv.threshold(opening_3, 10, 255, cv.THRESH_BINARY)
        cv.imshow('thresh', warstwa)
        cv.waitKey()
        szkielet = cv.ximgproc.thinning(warstwa, thinningType=cv.ximgproc.THINNING_GUOHALL)
        # cv.imshow('warstwa', gray)
        # cv.imshow('thinned', szkielet)
        # cv.waitKey()
        print('size', szkielet.size, result.size, warstwa.size, gray.size)
        result = cv.bitwise_or(result, szkielet)
    cv.imshow('Wynik', result)
    # cv.imwrite('Szkielet_49.jpg', result)
    cv.waitKey(0)
    return result



generacja_sladu(1)
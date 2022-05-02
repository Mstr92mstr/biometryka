import segmentation
import cv2 as cv
import numpy as np

#wpisanie pustej maski do wyniku operacji szkieletonizacji
result = cv.cvtColor(segmentation.segmentacja_maska[0], cv.COLOR_BGR2GRAY)

for n in range(1, segmentation.K):
    gray = cv.cvtColor(segmentation.segmentacja_maska[n], cv.COLOR_BGR2GRAY)
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
cv.imwrite('Szkielet1.jpg', result)
cv.waitKey(0)

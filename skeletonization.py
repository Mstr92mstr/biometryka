import segmentation
import cv2 as cv
import numpy as np

#wpisanie pustej maski do wyniku operacji szkieletonizacji
result = cv.cvtColor(segmentation.segmentacja_maska[0], cv.COLOR_BGR2GRAY)

for n in range(1, segmentation.K):
    gray = cv.cvtColor(segmentation.segmentacja_maska[n], cv.COLOR_BGR2GRAY)
    threshold, thresh = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)
    thinned = cv.ximgproc.thinning(thresh, thinningType=cv.ximgproc.THINNING_GUOHALL)
    print('size', thinned.size, result.size, thresh.size, gray.size)
    result = cv.bitwise_or(result, thinned)

cv.imshow('Wynik', result)
cv.waitKey(0)

import cv2 as cv
import numpy as np
from plantcv import plantcv as pcv

# Set global debug behavior to None (default), "print" (to file),
# or "plot" (Jupyter Notebooks or X11)
pcv.params.debug = "plot"


for i in range(32, 38):
    podbpis_zaklocony = 'base_of_sings/Szkielet' + str(i) + '.jpg'
    nazwa_zapis = 'signs_cleared/new_' + str(i) + '.jpg'

    img_zdj = cv.imread(podbpis_zaklocony)

    cv.imshow('zdjecie', img_zdj)
    cv.waitKey(0)

    kernel_3 = np.ones((1, 1), 'uint8')

    # closing_3 = cv.morphologyEx(img_zdj, cv.MORPH_CLOSE, kernel_3)
    # opening_3 = cv.morphologyEx(closing_3, cv., kernel_3)
    pruned_skeleton, segmented_img, segment_objects = pcv.morphology.prune(skel_img=img_zdj, size=2)
    cv.imshow('przed', img_zdj)
    cv.imshow('po', pruned_skeleton)
    cv.waitKey()
    # cv.imwrite(nazwa_zapis, opening_3)


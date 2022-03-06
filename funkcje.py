import cv2 as cv
import numpy as np


def usuniecie_tla(img):
    # threshold on blue (cold background)
    # Define lower and uppper limits of blue color
    lower = np.array([0, 0, 0])
    upper = np.array([255, 150, 150])
    # Create mask to only select background
    thresh = cv.inRange(img, lower, upper)
    # apply morphology
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
    morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    # invert morp image
    mask = 255 - morph
    # apply mask to image
    result = cv.bitwise_and(img, img, mask=mask)
    return result


def twarz_maska(img, faces_rect, blank):
    for (x, y, w, h) in faces_rect:
        radius = int(w*0.66)
        cent_x = int(x+w/2)
        cent_y = int(y+w/2)
        marked_face = cv.circle(img, (cent_x, cent_y), radius, (0, 0, 255), 1)
        #generacja maski
        mask = cv.circle(blank, (cent_x, cent_y), radius, 255, -1)
        return mask, marked_face

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
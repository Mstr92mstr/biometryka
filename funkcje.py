import cv2
import numpy as np


def usuniecie_tla(img):
    # threshold on blue (cold background)
    # Define lower and uppper limits
    lower = np.array([0, 0, 0])
    upper = np.array([255, 150, 150])
    # Create mask to only select black
    thresh = cv2.inRange(img, lower, upper)
    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # invert morp image
    mask = 255 - morph
    # apply mask to image
    result = cv2.bitwise_and(img, img, mask=mask)
    # save results
    cv2.imwrite('pills_result.jpg', result)
    return result

import cv2 as cv
import numpy as np

# kaskady = ['haar_face.xml', 'haarcascade_frontalface_alt.xml', 'haarcascade_frontalface_alt2.xml', 'haarcascade_frontalface_alt2_1.xml', 'haarcascade_frontalface_default.xml']
kaskady = ['haar_face.xml']
wyniki = np.zeros(len(kaskady))

for n in range(17):
    link = ('Resources/RAINBOW/'+str(11+n)+'.jpg')
    img = cv.imread(link)
    # cv.imshow('Twarze', img)
    for i in range(len(kaskady)):
        haar_cascade = cv.CascadeClassifier(kaskady[i])
        faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.04, minNeighbors=5)
        wyniki[i] += len(faces_rect)
        for (x, y, w, h) in faces_rect:
            cv.rectangle(img, (x, y), (x + w, y + w), (0, 0, 255), 2)
            cv.imshow(str(link), img)
            cv.waitKey(0)

for m in range(len(wyniki)):
    print('Dla palety RAINBOW i kaskady', kaskady[m], 'znaleziono', int(wyniki[m]), 'twarzy')

# for n in range(17):
#     link = ('Resources/Photos/ALL/GREY/'+str(11+n)+'.jpg')
#     img = cv.imread(link)
#     # cv.imshow('Twarze', img)
#     for i in range(len(kaskady)):
#         haar_cascade = cv.CascadeClassifier(kaskady[i])
#         faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=4)
#         wyniki[i] += len(faces_rect)
#
# for m in range(len(wyniki)):
#     print('Dla palety GREY i kaskady', kaskady[m], 'znaleziono', int(wyniki[m]), 'twarzy')
#
# for n in range(17):
#     link = ('Resources/Photos/ALL/YELLOW/'+str(11+n)+'.jpg')
#     img = cv.imread(link)
#     # cv.imshow('Twarze', img)
#     for i in range(len(kaskady)):
#         haar_cascade = cv.CascadeClassifier(kaskady[i])
#         faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=4)
#         wyniki[i] += len(faces_rect)
#
# for m in range(len(wyniki)):
#     print('Dla palety YELLOW i kaskady', kaskady[m], 'znaleziono', int(wyniki[m]), 'twarzy')
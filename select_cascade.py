import cv2 as cv

img = cv.imread('Resources/Photos/RAINBOW/41.jpg')
cv.imshow('Twarze', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

kaskady = ['haar_face.xml', 'haarcascade_frontalcatface.xml', 'haarcascade_frontalcatface_extended.xml', 'haarcascade_frontalface_alt.xml', 'haarcascade_frontalface_alt2.xml', 'haarcascade_frontalface_alt2_1.xml', 'haarcascade_frontalface_alt_tree.xml', 'haarcascade_frontalface_default.xml', 'haarcascade_profileface.xml']
wykryte = img

for i in range(len(kaskady)):
    haar_cascade = cv.CascadeClassifier(kaskady[i])
    faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=4)
    for (x, y, w, h) in faces_rect:
        cv.rectangle(wykryte, (x, y), (x + w, y + w), (0, 0, 255), 2)
        cv.imshow('Wykryto', wykryte)
    print('Liczba znalezionych twarzy :', len(faces_rect), 'kaskada', kaskady[i])
    wykryte = img
    cv.imshow('w petli', wykryte)
    cv.waitKey(0)

cv.waitKey(0)
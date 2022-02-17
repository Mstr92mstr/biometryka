import cv2 as cv

img = cv.imread('Resources/Photos/RAINBOW/32.jpg')
cv.imshow('Twarze', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# haar_cascade = cv.CascadeClassifier('haar_face.xml')
# haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt2_1.xml')
# faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)
# print('Liczba znalezionych twarzy : ', len(faces_rect))

kaskady = ['haar_face.xml', 'haarcascade_frontalcatface.xml', 'haarcascade_frontalcatface_extended.xml', 'haarcascade_frontalface_alt.xml', 'haarcascade_frontalface_alt2.xml', 'haarcascade_frontalface_alt2_1.xml', 'haarcascade_frontalface_alt_tree.xml', 'haarcascade_frontalface_default.xml', 'haarcascade_profileface.xml']
wykryte = cv.imread('Resources/Photos/01.jpg')

for i in range(len(kaskady)):
    haar_cascade = cv.CascadeClassifier(kaskady[i])
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=3)
    print('faces_rect', faces_rect)
    for (x, y, w, h) in faces_rect:
        cv.rectangle(wykryte, (x, y), (x + w, y + w), (0, 0, 255), 2)
        cv.imshow('Wykryto', wykryte)

    print('Liczba znalezionych twarzy :', len(faces_rect), 'kaskada', kaskady[i])
    wykryte = img
    cv.imshow('w petli', wykryte)
    cv.waitKey(0)

# for (x,y,w,h) in faces_rect:
#     cv.rectangle(img, (x,y), (x+w, y+w), (0,0,255), 2)
    # print(x,y,w,h, '(x,y,w,h)')
# cv.imshow('Twarz', img)
# cv.imshow('po wszystkim', img)
# resized = cv.resize(img, (500,500), interpolation=cv.INTER_AREA)
# cv.imshow('Datected', resized)

cv.waitKey(0)
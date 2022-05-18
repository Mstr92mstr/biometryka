import cv2 as cv

kaskady = ['haar_face.xml', 'haarcascade_frontalface_alt.xml', 'haarcascade_frontalface_alt2.xml', 'haarcascade_frontalface_alt_tree.xml', 'haarcascade_frontalface_default.xml', 'haarcascade_profileface.xml']
wyniki = [0, 0, 0, 0, 0, 0]

for n in range(26):
    nazwa_sciezki = ('Resources/Photos/RAINBOW/'+str(32+n)+'.jpg')
    img = cv.imread(nazwa_sciezki)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow('Twarze', img)
    for i in range(len(kaskady)):
        haar_cascade = cv.CascadeClassifier(kaskady[i])
        faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.03, minNeighbors=3)
        wyniki[i] += len(faces_rect)

for m in range(len(wyniki)):
    print('Dla kaskady', kaskady[m], 'znaleziono', wyniki[m], 'twarzy spośród', n, 'plików')




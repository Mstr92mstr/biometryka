import face_extract
from funkcje import *

img = face_extract.masked

twoDimage = img.reshape((-1,3))
twoDimage = np.float32(twoDimage)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# liczba ekstraktowanych kolorow do segmentacji
K = 4
attempts = 10
segmentacja_maska = []

ret, label, center = cv.kmeans(twoDimage, K, None, criteria, attempts, cv.KMEANS_PP_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
image_segmentation = res.reshape(img.shape)

cv.imshow('Twarz po segmentacji', image_segmentation)
n = 0
for kolor in center:
    print('kolor', kolor)
    segmentacja_maska.append(podzial_segmentacja(image_segmentation, kolor))

for n in range(K):
    tekst = ('maska ') + str(n)
    cv.imshow(tekst, segmentacja_maska[n])
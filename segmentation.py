import face_extract
from funkcje import *

img = face_extract.po_maskowaniu

twoDimage = img.reshape((-1,3))
twoDimage = np.float32(twoDimage)
# cv.imshow('twoDimage', twoDimage)
kryteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# liczba ekstraktowanych kolorow do segmentacji
K = 4
podejscia = 10
segmentacja_maska = []

ret, label, center = cv.kmeans(twoDimage, K, None, kryteria, podejscia, cv.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
image_segmentation = res.reshape(img.shape)

# cv.imshow('Twarz po segmentacji', image_segmentation)
n = 0
for kolor in center:
    print('kolor', kolor)
    segmentacja_maska.append(podzial_segmentacja(image_segmentation, kolor))

for n in range(K):
    tekst = ('maska ') + str(n)
    # cv.imshow(tekst, segmentacja_maska[n])
    kernel_3 = np.ones((2, 2), 'uint8')
    closing_3 = cv.morphologyEx(segmentacja_maska[n], cv.MORPH_CLOSE, kernel_3)
    opening_3 = cv.morphologyEx(closing_3, cv.MORPH_OPEN, kernel_3)
    tekst_2 = tekst + 'po morfologi'
    # cv.imshow(tekst_2, opening_3)

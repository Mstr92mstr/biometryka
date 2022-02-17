import cv2 as cv

# img = cv.imread('Resources/Photos/cat_large.jpg')
# cv.imshow('Cat', img)


def rescaleFrame(frame1, scale=0.75):
    width = int(frame1.shape[1] * scale)
    height = int(frame1.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame1, dimensions, interpolation=cv.INTER_AREA)


capture = cv.VideoCapture('Resources/Videos/dog.mp4')

while True:
    isTrue, frame = capture.read()

    frame_resized = rescaleFrame(frame, 0.2)

    cv.imshow('Video', frame)
    cv.imshow('Video skalowane', frame_resized)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()

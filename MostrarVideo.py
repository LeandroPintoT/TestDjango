import cv2 as cv
import numpy as np

video = cv.VideoCapture("prueba_imagenes.mp4")
while video.isOpened():
    ret, frame = video.read()
    cv.imshow("Video", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()

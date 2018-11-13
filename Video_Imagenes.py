import cv2
import os
from Rotar_Imagen import rotar_imagen


f = 1
while True:
    try:
        os.mkdir("f" + str(f))
        archivo = raw_input("ingrese el nombre del archivo: ")

        vidcap = cv2.VideoCapture(archivo)
        success, image = vidcap.read()
        count = 0
        success = True
        while success:
            cv2.imwrite("f" + str(f) + "/frame" + str(count) + ".jpg", image)  # save frame as JPEG file
            #os.system("Rotar_imagen.py f" + str(f) + " frame" + str(count))
            cv2.waitKey(2)
            rotar_imagen("f" + str(f), "frame" + str(count))
            os.remove("f" + str(f) + "/frame" + str(count) + ".jpg")
            success, image = vidcap.read()
            print 'Read a new frame: ', success
            count += 1
        break
    except OSError:
        f = f + 1

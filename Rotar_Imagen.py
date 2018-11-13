import cv2
import numpy as np
from PIL import Image
import sys


def rotar_imagen(carpeta, nombreFoto):
    imagen = Image.open(carpeta + "/" + nombreFoto + ".jpg")
    pixeles = imagen.load()
    img = imagen.transpose(Image.ROTATE_270)
    img.save(carpeta + "/" + nombreFoto + "new.jpg")

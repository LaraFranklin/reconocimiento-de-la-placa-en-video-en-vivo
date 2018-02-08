from __future__ import print_function

import cv2 as cv
import numpy as np
import urllib.request
import socket

# local modules
import time
from common import clock, draw_str

# contador
global capturas

# variables estaticas
ipServidor = "127.0.0.1" # Ip del servidor
puertoServidor = "12345" # Puerto del servidor para los socket
urlCamara = 'http://192.168.0.103:8080/shot.jpg'

# cordenadas de la placa
def cordenadas(cordenadas):
    x = [cordenadas[0][0][1], cordenadas[2][0][1]]
    y = [cordenadas[3][0][0], cordenadas[1][0][0]]
    return sorted(x), sorted(y)

# Envio de la URL al servidor
def envioSocket(url):
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect((ipServidor, puertoServidor))  # Ip del servidor y puerto
    soc.send(url.encode("utf8"))
    result_bytes = soc.recv(4096)
    return result_bytes.decode("utf8")


def identificarPlaca(img, capturas):
    # RGB to Gray scale conversion
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # Eliminación de ruido con filtro bilateral iterativo (elimina el ruido mientras conserva los bordes)
    noise_removal = cv.bilateralFilter(img_gray, 9, 75, 75)

    # Ecualización de histograma para obtener mejores resultados
    equal_histogram = cv.equalizeHist(noise_removal)

    # Apertura morfológica con un elemento de estructura rectangular
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    morph_image = cv.morphologyEx(equal_histogram, cv.MORPH_OPEN, kernel, iterations=15)

    # Resta de la imagen (Restando la imagen Morphed del histograma Imagen ecualizada)
    sub_morp_image = cv.subtract(equal_histogram, morph_image)

    # Umbral de la imagen
    ret, thresh_image = cv.threshold(sub_morp_image, 0, 255, cv.THRESH_OTSU)

    # Aplicación de detección de Canny Edge
    canny_image = cv.Canny(thresh_image, 250, 255)
    canny_image = cv.convertScaleAbs(canny_image)

    # dilation to strengthen the edges
    kernel = np.ones((3, 3), np.uint8)
    # Creating the kernel for dilation
    dilated_image = cv.dilate(canny_image, kernel, iterations=1)

    # Finding Contours in the image based on edges
    new, contours, hierarchy = cv.findContours(dilated_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]
    # Sort the contours based on area ,so that the number plate will be in top 10 contours
    screenCnt = None
    # loop over our contours
    for c in contours:
        # approximate the contour
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.06 * peri, True)  # Approximating with 6% error
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:  # Select the contour with 4 corners
            screenCnt = approx
            break

    try:
        x, y = cordenadas(screenCnt)
        if len(x) != 0 and len(y) != 0 and x[1].isdigit and x[0].isdigit and y[0].isdigit and y[1].isdigit:
            placa = img[x[0]:x[1], y[0]:y[1]]
            # placa = img[screenCnt[0][0][1]:screenCnt[2][0][1], screenCnt[3][0][0]:screenCnt[1][0][0]]
            path = "placas/placa-%d.png" % capturas
            cv.imwrite(path, placa)
            if (envioSocket(path)):
                time.sleep(3)
    except AttributeError:
        pass
    except:
        pass


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.4, minNeighbors=2, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


def draw_rects(img, rects, color):
    global capturas
    for x1, y1, x2, y2 in rects:
        capturas += 1
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv.imwrite("capturas/captura-%d.jpg" % capturas, img)
        identificarPlaca(img, capturas)


def videoLive():

    global capturas
    capturas = 9000
    args = {}
    cascade_fn = args.get('--cascade', "haarcascades/cars3.xml")
    cascade = cv.CascadeClassifier(cascade_fn)
    while True:
        imgResp = urllib.request.urlopen(urlCamara)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        img = cv.imdecode(imgNp, -1)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        t = clock()
        rects = detect(gray, cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        dt = clock() - t
        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt * 1000))
        cv.imshow('Detecta Vehiculo', vis)

        if cv.waitKey(5) == 27:
            break


if __name__ == '__main__':
    print("Inicia el proceso")
    videoLive()
    cv.destroyAllWindows()

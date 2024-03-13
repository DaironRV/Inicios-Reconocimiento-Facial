import cv2 as cv
import os
import imutils

modelo = 'fotos de Dalas'

ruta1='D:/Python/ReconocimientoFacial' 
rutacompleta = ruta1+ '/' + modelo

if not os.path.exists(rutacompleta):
    os.makedirs(rutacompleta)

ruidos = cv.CascadeClassifier('D:\Python\entrenamientos Open cv\opencv-4.x\data\haarcascades\haarcascade_frontalface_default.xml') 

camara = cv.VideoCapture(0)

id = 0

while True: 
    respuesta, captura = camara.read()
    
    if respuesta == False: 
        break
    
    captura = imutils.resize(captura, width=640)
    
    
    girces = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    idCamptura = captura.copy()
    
    caras = ruidos.detectMultiScale(girces, 1.27, 5)
    
    for (x, y, e1,e2 ) in  caras:
        cv.rectangle(captura,(x,y),(x+e1,y+e2), (255, 0 , 0), 1)
        rostroCapturado = idCamptura[y:y+e2, x:x+e1]
        rostroCapturado = cv.resize(rostroCapturado, (160, 160), interpolation=cv.INTER_CUBIC)
        cv.imwrite(rutacompleta+ '/imagen_{}.jpg'.format(id), rostroCapturado)
        id = id+1
    
    cv.imshow("resultado", captura)
    
    if id == 351:
        break
    
camara.release()
cv.destroyAllWindows()
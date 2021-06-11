import cv2
import sys
#print('bien')

# Get user supplied values
imagePath =  "foto.jpg"#"abba.png" #sys.argv[1]
cascPath =  "haarcascade_frontalface_default.xml" #sys.argv[2]

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale( #funcion para detectar objetos
    gray, #escala de grises
    scaleFactor=1.1, #compensa la escala tamaño de la cara
    minNeighbors=10, #declara cuantos objetos son detectados 
    minSize=(30,30), #
    flags = cv2.CASCADE_SCALE_IMAGE
)

#funcion que retorna una lista de rectangulos la cual cree que encontro una cara

print("Found {0} faces!".format(len(faces)))
# Draw a rectangle around the faces
for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
#la funcion anterior devuelve la ubicacion x e y y su ancho w y alto h

cv2.imshow("Faces Show",image)
cv2.waitKey(0)
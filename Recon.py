from passporteye import read_mrz
import pytesseract as pt
import cv2
import sys
import os
import pymysql
import base64
from PIL import Image
import io 
from simple_facerec import SimpleFacerec
import uuid
import streamlit as st


connection = pymysql.connect(host="localhost",user="root",passwd="",database="pythontest" )
cursor = connection.cursor()

cam = cv2.VideoCapture(0)
 
cv2.namedWindow("Verification")
 
img_counter = 0
 
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)
 
    k = cv2.waitKey(1)
    if k%256 == 27:
        
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        
        img_name = "opencv_frame.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        cam.release()


cv2.destroyAllWindows()
def MRZ():

 pt.pytesseract.tesseract_cmd=(r'C:\Users\Yosr AROUI\tesseract.exe')
 print("paravoce")

 mrz = read_mrz(r'C:\Users\Yosr AROUI\Documents\KPMG\sourcecode\opencv_frame.jpg')
 imagePath = (r'C:\Users\Yosr AROUI\Documents\KPMG\sourcecode\opencv_frame.jpg')

 image = cv2.imread(imagePath)
 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

 faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
 faces = faceCascade.detectMultiScale(
    gray,
    
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(50, 50)
 )

 print("[INFO] Found {0} Faces.".format(len(faces)))

 for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi_color = image[y:y + h, x:x + w]
    print("[INFO] Object found. Saving locally.")
    imgg=cv2.imwrite(('images/' +str(w) + str(h)  + '_faces.jpg'), roi_color)

 status = cv2.imwrite('faces_detected.jpg', image)
 print("[INFO] Image faces_detected.jpg written to filesystem: ", status)

 if mrz == None:
    print("Invalid document")
 else:
    mrz_data = mrz.to_dict()
    print('Nationality :'+ mrz_data['country'])
    print(type (mrz_data['country']))
    print('Name :'+ mrz_data['names'])
    print('Surname :'+ mrz_data['surname'])
    print('passportType :' + mrz_data['type'])
    print('DateofBirth :' + mrz_data['date_of_birth'])
    print('Gender :' + mrz_data['sex'])
    print('Expiration date :' + mrz_data['expiration_date'])
    #print('ID Number :' + mrz_data['personal_number'])
    print('Passport number  :' + mrz_data['number'])
    cv2.imwrite(('images/' + mrz_data['names'] + '_faces.jpg'), imgg)
    X=mrz_data['country']
    Y=mrz_data['names']
    Z=mrz_data['surname']
    file = open(r'C:\Users\Yosr AROUI\Documents\KPMG\sourcecode\images\1_faces.jpg','rb').read()
    file = base64.b64encode(file)
    print(type(file))
    id = uuid.uuid4()
    idd=id.bytes
    V=file+idd

 query="INSERT INTO user (Nationality,name,Surname,image) VALUES (%s,%s,%s,%s)"
 value=(X,Y,Z,V)
 cursor = connection.cursor()
 cursor.execute(query,value)
 connection.commit()
 def run():
    Mrz()

"""
# Encode faces from a foldr
sfr = SimpleFacerec()
sfr.load_encoding_images(r'.\images')

# Load Camera
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    # Detect Faces
    print("frame main",frame)
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
"""
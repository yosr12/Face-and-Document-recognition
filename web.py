import cv2
import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import base64
from passporteye import read_mrz
import pytesseract as pt
import os
import matplotlib.image as mpimg
import face_recognition
import subprocess
import random 
import questions
import imutils
import f_liveness_detection
from imutils.video import VideoStream
from simple_facerec import SimpleFacerec
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import pymysql
import uuid


pic=Image.open(r"C:\Users\Yosr AROUI\Downloads\kpmg.png")
st.set_page_config(page_title="KPMG", page_icon=pic, layout="centered", initial_sidebar_state = "auto")

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.i = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        i =self.i+1
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (95, 207, 30), 3)
            #cv2.rectangle(img, (x, y - 40), (x + w, y), (95, 207, 30), -1)
            cv2.putText(img,str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            roi_color = img[y:y + h, x:x + w]
            print("[INFO] Object found. Saving locally.")
            cv2.imwrite(('images/' +'_faces.jpg'), roi_color)


        return img



   
def get_image_download_link(img,filename,text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href =  f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href
def face_detect(image,sf,mn):
    i = 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,sf,mn)
    for (x, y, w, h) in faces:
        i = i+1
        cv2.rectangle(image, (x, y), (x + w, y + h), (237, 30, 72), 3)
        cv2.rectangle(image, (x, y - 40), (x + w, y),(237, 30, 72) , -1)
        cv2.putText(image, 'F-'+str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    resi_image = cv2.resize(image, (350, 350))
    return resi_image,i,image


def Mrz():
    img_file = st.camera_input("Webcam image")
    if img_file:
        with open ('images/test.jpg','wb') as file:
            file.write(img_file.getbuffer())
            
    pt.pytesseract.tesseract_cmd=(r'C:\Users\Yosr AROUI\tesseract.exe')
    print("paravoce")

    mrz = read_mrz(r'images\test.jpg')
    imagePath = (r'images\test.jpg')
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
    connection = pymysql.connect(host="localhost",user="root",passwd="",database="pythontest")
    cursor = connection.cursor()
    if mrz != None:
        print("valid document")
        print(mrz)
        requeteee = """Select * FROM user where mrz Like %s"""
        cursor.execute(requeteee, (mrz))
        SQLResult = cursor.fetchone()
        print(SQLResult)
        if SQLResult != None :
            print("user exist")       
        elif SQLResult == None :
            uid=uuid.uuid1()
            S_uid = str(uid)
            mrz_data = mrz.to_dict()
            print('Nationality :'+ mrz_data['country'])
            print('Name :'+ mrz_data['names'])
            print('Surname :'+ mrz_data['surname'])
            print('passportType :' + mrz_data['type'])
            print('DateofBirth :' + mrz_data['date_of_birth'])
            print('Gender :' + mrz_data['sex'])
            print('Expiration date :' + mrz_data['expiration_date'])
            #print('ID Number :' + mrz_data['personal_number'])
            print('Passport number  :' + mrz_data['number'])
            st.write('Name :',mrz_data['names'])
            st.write('Surname :',mrz_data['surname'])
            st.write('Nationality:',mrz_data['country'])
            st.write('Date Of Birth :',mrz_data['date_of_birth'] )
            st.write('Gender :', mrz_data['sex'])
            st.write('Type :',mrz_data['type'])
            Country = mrz_data['country'] 
            Name = mrz_data['names']
            Surname = mrz_data['surname']
            Type =  mrz_data['type']
            Gender =  mrz_data['sex']
            Birth_Date = mrz_data['date_of_birth']
            Img_mrz = 'images/'+S_uid+'.jpg'
            query="INSERT INTO user (mrz,Nationality,name,Surname,type,gender,birth_date,image) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
            value = (mrz,Country,Name,Surname,Type,Gender,Birth_Date,Img_mrz)
            connection.cursor()
            cursor.execute(query,value)
            connection.commit()
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_color = image[y:y + h, x:x + w]
            print("[INFO] Object found. Saving locally.")
            cv2.imwrite(('images/' +'1_faces.jpg'), roi_color)

            cv2.imwrite(('images/'+S_uid+'.jpg'), roi_color)
            
            status = cv2.imwrite('faces_detected_Passport.jpg', image)
            print("[INFO] Image faces_detected.jpg written to filesystem: ", status)            
    if mrz == None:
        print("Invalid document")
        

def Verif():
     img = cv2.imread(r'images\_faces.jpg')
     rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     img_encoding = face_recognition.face_encodings(rgb_img)[0]

     img2 = cv2.imread(r'images\1_faces.jpg')
     rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
     img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

     result = face_recognition.compare_faces([img_encoding], img_encoding2)
     print("Result: ", result[0])
     st.image(rgb_img)
     st.image(rgb_img2)
    
     #st.subheader(result[0])
     if result [0]== True :
        cv2.imwrite('images/' + 'verif.jpg',img)
        st.markdown(
                 '''<h4 style='text-align: left; color: #008000;'> TRUE</h4>''',unsafe_allow_html=True) 
     elif result [0] == False:
       os.remove(r'images\_faces.jpg')
       st.markdown(
                 '''<h4 style='text-align: left; color: #d73b5c;'> False</h4>''',unsafe_allow_html=True) 

       

       
def show_image(cam,text,color = (0,0,255)):
    ret, im = cam.read()
    im = imutils.resize(im, width=720)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    #im = cv2.flip(im, 1)   
    for (x, y, w, h) in faces:
     cv2.putText(im,text,(10,50),cv2.FONT_HERSHEY_COMPLEX,1,color,2)
     cv2.rectangle(im, (x, y), (x + w, y + h), (95, 207, 30), 3)

     #roi_color = im[y:y + h, x:x + w]
     #print("[INFO] Object found. Saving locally.")
     #cv2.imwrite(('images/' +'L_faces.jpg'), roi_color)

    return im
def show_YY(cam):
    ret, im = cam.read()
    im = imutils.resize(im, width=720)
    return im 
def Matching():
    sfr = SimpleFacerec()
    sfr.load_encoding_images("images/")
    #@st.cache(allow_output_mutation=True)
    #def get_cap():
     #   return cv2.VideoCapture(0)
    #cam = get_cap()
    #frameST = st.empty() 
    #while True:      
       # im = show_YY(cam)
        #frameST.image(im, channels="BGR")   
#        if cv2.waitKey(1) &0xFF == ord('q'):
#            cam.release()

       # frame = show_YY(cam)
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        # Detect Faces
        face_locations, face_names = sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
    else:
        st.write('Stopped')
        #cv2.imshow("Frame", frame)

      


def Liveness():
  #  vs = VideoStream(src=0).start()
    @st.cache(allow_output_mutation=True)
    def get_cap():
        return cv2.VideoCapture(0)
    cam = get_cap()
    frameST = st.empty()        
    # parameters 
    COUNTER, TOTAL = 0,0
    counter_ok_questions = 0
    counter_ok_consecutives = 0
    limit_consecutives = 1
    limit_questions = 6
    counter_try = 0
    limit_try = 50


    for i_questions in range(0,limit_questions):
        index_question = random.randint(0,5)
        question = questions.question_bank(index_question)
        im = show_image(cam,question)
        frameST.image(im, channels="BGR")    

        if cv2.waitKey(1) &0xFF == ord('q'):
            break 

        for i_try in range(limit_try):
            ret, im = cam.read()
            im = imutils.resize(im, width=720)
            im = cv2.flip(im, 1)
            TOTAL_0 = TOTAL
            out_model = f_liveness_detection.detect_liveness(im,COUNTER,TOTAL_0)
            TOTAL = out_model['total_blinks']
            COUNTER = out_model['count_blinks_consecutives']
            dif_blink = TOTAL-TOTAL_0
            if dif_blink > 0:
                blinks_up = 1
            else:
                blinks_up = 0

            challenge_res = questions.challenge_result(question, out_model,blinks_up)

            im = show_image(cam,question)
         #   cv2.imshow('liveness_detection',im)
            frameST.image(im, channels="BGR")
            if cv2.waitKey(1) &0xFF == ord('q'):
                break 

            if challenge_res == "pass":
                im = show_image(cam,question+" : ok")
               # cv2.imshow('liveness_detection',im)
                frameST.image(im, channels="BGR")
                if cv2.waitKey(1) &0xFF == ord('q'):
                    break

                counter_ok_consecutives += 1
                if counter_ok_consecutives == limit_consecutives:
                    counter_ok_questions += 1
                    counter_try = 0
                    counter_ok_consecutives = 0
                    break
                else:
                    continue

            elif challenge_res == "fail":
                counter_try += 1
                show_image(cam,question+" : fail")
            elif i_try == limit_try-1:
                break
        frameST.image(im, channels="BGR")    

        if counter_ok_questions ==  limit_questions:
            while True:
                im = show_image(cam,"LIVENESS SUCCESSFUL",color = (0,180,0))
                #cv2.imshow('liveness_detection',im)
                frameST.image(im, channels="BGR")
                if cv2.waitKey(1) &0xFF == ord('q'):
                    break
        elif i_try == limit_try-1:
            while True:
                im = show_image(cam,"LIVENESS FAIL")
               # cv2.imshow('liveness_detection',im)
                frameST.image(im, channels="BGR")
                if cv2.waitKey(1) &0xFF == ord('q'):
                    break
            break 

        else:
            continue    
        
    cv2.destroyAllWindows()
        

    


    

def run():
    st.title("KPMG")
    activities = ["Face","MRZ","Verification","Liveness","Matching"]
    #st.set_option('deprecation.showfileUploaderEncoding', False)
    
    choice = st.selectbox("Choose among the given options:", activities)
    st.write('You selected:', choice)

    
    #if choice == 'Image':
       # st.markdown(
           # '''<h4 style='text-align: left; color: #d73b5c;'>*</h4>''',
           # unsafe_allow_html=True)
       # img_file = st.file_uploader("Choose an Image", type=['jpg', 'jpeg', 'jfif', 'png'])
       # if img_file is not None:
         #   img = np.array(Image.open(img_file))
          #  img1 = cv2.resize(img, (350, 350))
           # place_h = st.columns(2)
          #  place_h[0].image(img1)
          #  st.markdown(
            #    '''<h4 style='text-align: left; color: #d73b5c;'>* Increase & Decrease it to get better accuracy.</h4>''',
            #    unsafe_allow_html=True)
           # scale_factor = st.slider("Set Scale Factor Value", min_value=1.1, max_value=1.9, step=0.10, value=1.3)
          #  min_Neighbors = st.slider("Set Scale Min Neighbors", min_value=1, max_value=9, step=1, value=5)
          #  fd, count, orignal_image = face_detect(img, scale_factor, min_Neighbors)
          #  place_h[1].image(fd)
          #  if count == 0:
          #      st.error("No People found!!")
          #  else:
             #   st.success("Total number of People : " + str(count))
              #  st.markdown(get_image_download_link(result, img_file.name, 'Download Image'), unsafe_allow_html=True)
                
    if choice == 'Face':
            st.warning('Keep a neutral facial expression, or a natural smile,and remember to have both eyes open. 2-Face the camera directly with full face in view.')

            st.markdown(
                 '''<h4 style='text-align: left; color: #d73b5c;'> 1-Face Detection</h4>''',
                 unsafe_allow_html=True)
           
            webrtc_streamer(key="example", video_processor_factory=VideoTransformer)
       
    if choice == 'MRZ':
        st.warning('Create proper lighting conditions')

        st.markdown( '''<h4 style='text-align: left; color: #d73b5c;'> 2-Put your passport in front of the camera </h4>''', unsafe_allow_html=True)

        Mrz()

    if choice == 'Verification':
        
        st.markdown( '''<h4 style='text-align: left; color: #d73b5c;'> 3-Verfication </h4>''', unsafe_allow_html=True)
        Verif()
    if choice == 'Liveness':
        st.markdown( '''<h4 style='text-align: left; color: #d73b5c;'> 4-Liveness </h4>''', unsafe_allow_html=True)
        Liveness() 
    if choice == 'Matching':
        st.markdown(
            '''<h4 style='text-align: left; color: #d73b5c;'>*</h4>''',
            unsafe_allow_html=True)
        img_file = st.file_uploader("Choose an Image", type=['jpg', 'jpeg', 'jfif', 'png'])   
        if img_file is not None:
            img = np.array(Image.open(img_file))
            img1 = cv2.resize(img, (350, 350)) 
            place_h = st.columns(2)
            place_h[0].image(img1)
        Matching()
        

        
run()
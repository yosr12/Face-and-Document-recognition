import config as cfg
import dlib
import cv2
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist



class eye_blink_detector():
    def __init__(self):
      
        self.predictor_eyes = dlib.shape_predictor(cfg.eye_landmarks)

    def eye_blink(self,gray,rect,COUNTER,TOTAL):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
       
        shape = self.predictor_eyes(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)
       
        ear = (leftEAR + rightEAR) / 2.0
       
        if ear < cfg.EYE_AR_THRESH:
            COUNTER += 1
       
        else:
          
            if COUNTER >= cfg.EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            COUNTER = 0
        return COUNTER,TOTAL

    def eye_aspect_ratio(self,eye):
      
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
    
        C = dist.euclidean(eye[0], eye[3])
      
        ear = (A + B) / (2.0 * C)
        
        return ear

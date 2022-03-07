import  numpy as np
import cv2

feed=cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret,frame = feed.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)

        eyes= eye_cascade.detectMultiScale(gray[y:y+w,x:x+w],1.3,5)

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(frame[y:y+h,x:x+w],(ex,ey),(ex+ew,ey+eh),(0,255,0),5)

    cv2.imshow('frame',frame)

    if(cv2.waitKey(1)==ord('q')):
        break

feed.release()
cv2.destroyAllWind
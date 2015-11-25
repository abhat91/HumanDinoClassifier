import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

ret, frame = cap.read()
while ret:


    #img = cv2.imread('sachin.jpg')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.ellipse(frame,(x+w/2,y+h/2),((w)/2,(h)/2),0,0,360,(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.ellipse(roi_color,(ex+ew/2,ey+eh/2),((ew)/2,(eh)/2),0,0,360,(0,255,0),2)
    cv2.imshow("w1", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()

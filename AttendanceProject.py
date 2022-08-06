import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime,date
from playsound import playsound
with open('Attendance.csv','r+') as f:
 f.truncate()
 f.writelines('Name,Date,Time')
path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
 curImg = cv2.imread(f'{path}/{cl}')
 images.append(curImg)
 classNames.append(os.path.splitext(cl)[0])
print(classNames)
 
def findEncodings(images):
 encodeList = []
 for img in images:
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  encode = face_recognition.face_encodings(img)[0]
  encodeList.append(encode)
 return encodeList
 
def markAttendance(name,img):
 with open('Attendance.csv','r+') as f:
  myDataList = f.readlines()
  nameList = []
  for line in myDataList:
   entry = line.split(',')
   nameList.append(entry[0])
  if name not in nameList:
   now = datetime.now()
   dtString = now.strftime('%H:%M:%S')
   today=date.today()
   f.writelines(f'\n{name},{today},{dtString}')
   if name=='AKHIL':
    cv2.imwrite('detectedimages/akhil.jpg',img)
    playsound('Akhil.mp3')
   elif name=='NEEVEA':
    cv2.imwrite('detectedimages/neevea.jpg',img)
    playsound('Neevea.mp3')
   elif name=='ELON MUSK':
    cv2.imwrite('detectedimages/elon musk.jpg',img)
    playsound('Elon musk.mp3')
   elif name=='JACK MA':
    cv2.imwrite('detectedimages/jack ma.jpg',img)
    playsound('Jack ma.mp3')
   elif name=='BILL GATES':
    cv2.imwrite('detectedimages/bill gates.jpg',img)
   elif name=='ATHIRA':
    cv2.imwrite('detectedimages/athira.jpg',img)
    playsound('Athira.mp3')
   elif name=='LEKSHMI':
    cv2.imwrite('detectedimages/lekshmi.jpg',img)
    playsound('Lekshmi.mp3')
 
encodeListKnown = findEncodings(images)
print('Encoding Complete')
 
cap = cv2.VideoCapture(0)
 
while True:
 success, img = cap.read()
 imgS = cv2.resize(img,(0,0),None,0.25,0.25)
 imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
 
 facesCurFrame = face_recognition.face_locations(imgS)
 encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
 
 for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
   matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
   faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
   
   matchIndex = np.argmin(faceDis)   
   if matches[matchIndex]:
    name = classNames[matchIndex].upper()
    
    y1,x2,y2,x1 = faceLoc
    y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
    cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    markAttendance(name,img)
   
 cv2.imshow('Webcam',img)
 key=cv2.waitKey(1)
 if key==27:
   cap.release()
   cv2.destroyAllWindows()
   break
      

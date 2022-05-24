import cv2
import numpy as np
import os
import face_recognition_models
from datetime import datetime

path='images';
images=[];
personName=[];
myList=os.listdir(path);
print(myList);
for cu_img in myList:
    current_Img=cv2.imread(f'{path}/|{cu_img}');
    images.append(current_Img);
    personName.append(os.path.splitext(cu_img)[0])

print(personName);

def faceEncoding(images):
    encodeList = []
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodesListKnown=faceEncoding(images)
print(faceEncoding(images));

print("All Encoding Complete: ");
def Register(name):
    with open('Attandance_Sheet.csv','r+') as f:
        myData=f.readlines()
        nameList=[]
        for line in myData:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now=datetime.now()
            tStr=time_now.strftime('%H:%M:%S')
            dStr=time_now.strftime('%d/%m/%Y')
            f.writelines(f'{name},{tStr},{dStr}')
cam=cv2.VideoCapture(0)
while True:
    ret,frame=cam.read();
    faces=cv2.resize(frame,(0,0),None,0.25,0.25);
    faces=cv2.cvtColor(faces,cv2.COLOR_BGR2GRAY);

    facesCurrent=face_recognition_models.face_recognition_model_location(faces);
    EncodesCurrentFrame=face_recognition_models.face_encodings(faces,facesCurrent);
    for Encodeface,faceLoc in zip(EncodesCurrentFrame,facesCurrent):
        match=face_recognition_models.compare_faces(encodesListKnown,Encodeface);
        faceDis=face_recognition_models.face_distance(encodesListKnown,Encodeface);

        matchIndex=np.argmin(faceDis);

        if match[matchIndex]:
            name=personName[matchIndex].upper();
            # print(name)
            x1,x2,y1,y2=faceLoc
            y1,x2,y2,x1=y1*4,y2*4,y2*4,x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
            Register(name)

    cv2.imshow("Cemera",frame)
    if cv2.waitKey(10)==13:
        break

# cap.release()
cv2.destroyAllWindows()
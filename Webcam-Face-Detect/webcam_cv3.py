import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
from keras.models import model_from_json
import numpy as np


cascPath = "Webcam-Face-Detect/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='Webcam-Face-Detect/webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0


json_file = open('./fm.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./fm.h5")
print(loaded_model.summary())
class_names = ['with_mask', 'without_mask']
count = 0;
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(80, 80),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    if count > 10:
        cv2.putText(frame, "Mask Detection: PASS", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Mask Detection: NOT PASS", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    for (x, y, w, h) in faces:

        img = frame[y:y+h, x:x+w]
        print(frame)
        image_size = 112
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.array(img, dtype = 'uint8')
        img = cv2.resize(img, (image_size, image_size))
        img = img.reshape(image_size, image_size, 1)
        result = loaded_model.predict(np.array([img]))
        print(result)
        probab = max(result.flatten())
        result = (class_names[np.argmax(result)])

        print(probab)
        # print(result)
        if result == 'with_mask' :
            detection = result
            rec = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(rec, detection, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            count += 1
        else:
            detection = result
            rec = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(rec, detection, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            count -= 1


    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

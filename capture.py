import cv2
import time
import requests
import pyrebase
config = {
  "apiKey": "",
  "authDomain": "fire-detection-5c3f0.firebaseapp.com",
  "databaseURL": "https://fire-detection-5c3f0.firebaseio.com",
  "projectId": "fire-detection-5c3f0",
  "storageBucket": "fire-detection-5c3f0.appspot.com",
  "messagingSenderId": "150593024647",
  "appId": "1:150593024647:web:239f68368ac96bf24bbe22",
  "measurementId": "G-25S62FN0SW"
}
now = time.localtime()
capture = cv2.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 480)
daytime = "%04d-%02d-%02d-%02dh-%02dm-%02ds" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
frame_set = []
start_time = time.time()
while True:
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if time.time() - start_time >= 5: #<---- Check if 5 sec passed
        img_name = "detect_fire/fire_frame_{}{}".format(daytime,'.jpg')
        cv2.imwrite(img_name, frame)
        firebase= pyrebase.initialize_app(config)
        storage = firebase.storage()
        path_on_cloud = "images/fire_frame_{}{}".format(daytime,'.jpg')
        path_local = "detect_fire/fire_frame_{}{}".format(daytime,'.jpg')
        print("{} written!".format(daytime))
        storage.child(path_on_cloud).put(path_local)
        start_time = time.time()

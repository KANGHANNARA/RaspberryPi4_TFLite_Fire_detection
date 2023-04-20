import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import requests
import pyrebase
from datetime import datetime
import json
import motortest1

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

now = datetime.today().strftime("%Y%m%d_%H%M%S")
firebase= pyrebase.initialize_app(config)
fire_auth = firebase.auth()
fire_email = "rkdgksskfk2@naver.com"
fire_password = "feromon0309"
user = fire_auth.sign_in_with_email_and_password(fire_email, fire_password)
user_uid = fire_auth.get_account_info(user['idToken'])

img_name = "detect_fire/77.jpg"
#업로드할 파일명
uploadfile = img_name
#업로드할 새로운파일이름
#업로드할 파일의 확장자 구하기
s = os.path.splitext(uploadfile)[1]
filename = now + s
print(img_name + " written!")
storage = firebase.storage()
storage.child("fireimages/"+filename).put(uploadfile, user['idToken'])
fileUrl = storage.child("fireimages/"+filename).get_url(user['idToken']) #0은 저장소 위치 1은 다운로드 url 경로이다.
#동영상 파일 경로를 알았으니 어디에서든지 참조해서 사용할 수 있다.
#print (fileUrl)
#업로드한 파일과 다운로드 경로를 database에 저장하자. 
#save files info in database
db = firebase.database()
data = ""  #json.dumps(fileUrl)
results = db.child("fireimages").child(now).set(data, user['idToken'])
print("OK")

import requests
import pyrebase
import RPi.GPIO as GPIO
from time import sleep

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
firebase = pyrebase.initialize_app(config)
db = firebase.database()

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

#while True:
result = db.child("motor/act").get()
print("result:",result.val())
sleep(1)

GPIO.cleanup()
#result = firebase.get('das')
#print(result)

#fire_auth = firebase.auth()
#fire_email = "rkdgksskfk2@naver.com"
#fire_password = "feromon0309"
#user = fire_auth.sign_in_with_email_and_password(fire_email, fire_password)
#user_uid = fire_auth.get_account_info(user['idToken'])

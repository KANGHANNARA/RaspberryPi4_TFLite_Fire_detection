# Import packages
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
## 라즈베리파이 GPIO 패키지,firebase 패키지
import RPi.GPIO as GPIO
from time import sleep
# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
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
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x480')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu
firebase= pyrebase.initialize_app(config)
start_time = time.time()
now = datetime.today().strftime("%Y%m%d_%H%M%S")
fire_auth = firebase.auth()
fire_email = "rkdgksskfk2@naver.com"
fire_password = "feromon0309"
user = fire_auth.sign_in_with_email_and_password(fire_email, fire_password)
user_uid = fire_auth.get_account_info(user['idToken'])

GPIO.setwarnings(False)
# motor
# 모터 상태
STOP  = 0
FORWARD  = 1
BACKWORD = 2

# 모터 채널
CH1 = 0
CH2 = 1
                
# PIN 입출력 설정
OUTPUT = 1
INPUT = 0

# PIN 설정
HIGH = 1
LOW = 0

# 실제 핀 정의
#PWM PIN
ENA = 26  #37 pin
ENB = 0   #27 pin
IN5 = 18
IN6 = 12

#GPIO PIN
IN1 = 19  #37 pin
IN2 = 13  #35 pin
IN3 = 6   #31 pin
IN4 = 5   #29 pin

# GPIO 모드 설정 
GPIO.setmode(GPIO.BCM)

GPIO.setup(IN5, GPIO.OUT)
GPIO.setup(IN6, GPIO.OUT)
p= GPIO.PWM(IN5, 50)  #PMW:펄스 폭 변조
o= GPIO.PWM(IN6, 50)  #PMW:펄스 폭 변조

p.start(0)
o.start(0)
# 핀 설정 함수
def setPinConfig(EN, INA, INB):        
    GPIO.setup(EN, GPIO.OUT)
    GPIO.setup(INA, GPIO.OUT)
    GPIO.setup(INB, GPIO.OUT)
    # 100khz 로 PWM 동작 시킴 
    pwm = GPIO.PWM(EN, 100) 
    # 우선 PWM 멈춤.   
    pwm.start(0) 
    return pwm


#모터 핀 설정
#핀 설정후 PWM 핸들 얻어옴 
pwmA = setPinConfig(ENA, IN1, IN2)
pwmB = setPinConfig(ENB, IN3, IN4)

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        
        if((scores[i] >= 0.95) and (scores[i] <= 1.0)) and time.time() - start_time >= 5:
            img_name = "detect_fire/fire_frame_{}{}".format(now,'.jpg')
            cv2.imwrite(img_name, frame)
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
            #업로드한 파일을 database에 저장하자. 
            #save files info in database
            db = firebase.database()
            data = ""#json.dumps(fileUrl)
            results = db.child("fireimages").child(now).set(data, user['idToken'])
            print("OK")
            start_time = time.time()
            # firebase motor 값 가져오기
            motor = db.child("motor/act").get()
            print(motor.val())
            if motor.val() == 1:
                
                # 모터 제어 함수
                def setMotorContorl(pwm, INA, INB, speed, stat):

                    #모터 속도 제어 PWM
                    pwm.ChangeDutyCycle(speed)  
    
                    if stat == FORWARD:
                        GPIO.output(INA, motor.val())
                        GPIO.output(INB, 0)
        
                    #정지
                    elif stat == STOP:
                        GPIO.output(INA, 0)
                        GPIO.output(INB, 0)
        
                # 모터 제어함수 간단하게 사용하기 위해 한번더 래핑(감쌈)
                def setMotor(ch, speed, stat):
                    if ch == CH1:
                        #pwmA는 핀 설정 후 pwm 핸들을 리턴 받은 값이다.
                        setMotorContorl(pwmA, IN1, IN2, speed, stat)
                    else:
                    #pwmB는 핀 설정 후 pwm 핸들을 리턴 받은 값이다.
                        setMotorContorl(pwmB, IN3, IN4, speed, stat)
                #제어 시작
                # 앞으로 100프로 속도로
                setMotor(CH1, 25, FORWARD)
                setMotor(CH2, 100, FORWARD)
                #5초 대기
                sleep(5)

                #정지 
                setMotor(CH1, 80, STOP)
                setMotor(CH2, 80, STOP)
                
                p.ChangeDutyCycle(2.5) #최솟값
                o.ChangeDutyCycle(2.5) #최솟값
                time.sleep(1)
    
                p.ChangeDutyCycle(7.5) #0
                o.ChangeDutyCycle(7.5) #0
                time.sleep(1)
    
                p.ChangeDutyCycle(12.5) #최댓값
                o.ChangeDutyCycle(12.5) #최댓값
                time.sleep(1)
                # 종료
                #GPIO.cleanup()
                db = firebase.database()
                motor = db.child("motor").update({"act":"0"})
            
    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()

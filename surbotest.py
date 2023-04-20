import RPi.GPIO as GPIO
import time

GPIO.setwarnings(False)

GPIO.setmode(GPIO.BCM)
IN5 = 18
IN6 = 12
GPIO.setup(IN5, GPIO.OUT)
GPIO.setup(IN6, GPIO.OUT)

p= GPIO.PWM(IN5, 50)  #PMW:펄스 폭 변조
o= GPIO.PWM(IN6, 50)  #PMW:펄스 폭 변조

p.start(0)
o.start(0)


try:
    p.ChangeDutyCycle(2.5) #최솟값
    o.ChangeDutyCycle(2.5) #최솟값
    time.sleep(1)
    
    p.ChangeDutyCycle(7.5) #0
    o.ChangeDutyCycle(7.5) #0
    time.sleep(1)
    
    p.ChangeDutyCycle(12.5) #최댓값
    o.ChangeDutyCycle(12.5) #최댓값
    time.sleep(1)

    

except KeybordInterrupt:

     p.stop()

GPIO.cleanup()
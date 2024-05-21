import cv2
import time
import datetime
from twilio.rest import Client
import keys

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

detection = False
detection_stopped_time = None #record a bit certain time after someone leaves the frame also helps if our classifier suddenly stops
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

#how to save videos
frame_size =(int(cap.get(3)),int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")


while True:
    _,frame = cap.read()

    #adding live time
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, frame.shape[0] - 10)
    current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    cv2.putText(frame, current_time, position, font, 1, (255,255,255), 2, cv2.LINE_AA)

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)   #1.0 - 1.5 Lower is more accurate but lesser speed
    bodies = body_cascade.detectMultiScale(gray_image, 1.3, 5)

    if len(faces) + len(bodies) > 0:        # we detected a face
        if detection:   # if we already detected a face/body just say timer started is false
            timer_started = False
        else:       
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20.0, frame_size) #video name, 4char code, Fps, frame size
            print("Started Recording")

            client = Client(keys.account_sid, keys.auth_token)
            message = client.messages.create(
                body = "A person detected in the camera!",
                from_ = keys.twilio_number,
                to = keys.target_number
            )
            print("Message sent: "+message.body)


    elif detection :        #suddenly not detecting a frame
        if timer_started:       #check if timer was started
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:   #check if more than 5 seconds passed
                detection = False
                timer_started = False
                out.release()
                print("Stop Recording")

        else:               #timer not started, so start the timer
            timer_started = True
            detection_stopped_time = time.time()

    #writes the video file into host directory
    if detection:
        out.write(frame)
    #for (x,y, width, height) in faces:
    #    cv2.rectangle(frame, (x,y), (x+width, y+height),(255,0,0), 3)
        

    cv2.imshow("Camera",frame)
    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
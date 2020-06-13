import cv2, time, pandas as pd
from datetime import datetime

first_frame = None
status_list=[None,None]
times=[]
df = pd.DataFrame(columns=["Start","End"])
video = cv2.VideoCapture(0)

time.sleep(5) #this is needed for the webcame on my Macbook Pro

while True:

    check, frame = video.read()

    status = 0 #no motion in the current frame

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)

    #getting a firstframe to use for comparison in finding movement
    if first_frame is None:
        first_frame = gray
        print("FirstFrame:",' ')
        print(first_frame)
        continue

    #building the various frames
    delta_frame=cv2.absdiff(first_frame,gray)
    thresh_frame=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
    thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)

    #adding rectangles to moving objects
    (cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
        status=1 #motion found
        (x, y, w, h)=cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),3)

    #finds the times that an object entered and exited the frame
    status_list.append(status) 
    status_list = status_list[-2:]
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())

    #creating the windows
    cv2.imshow("GrayFrame",gray)
    cv2.imshow("DeltaFrame",delta_frame)
    cv2.imshow("ThreshFrame",thresh_frame)
    cv2.imshow("ColourFrame",frame)

    key=cv2.waitKey(1)
    

    #escapes the look
    if key==ord("q"):
        if status==1:
            times.append(datetime.now())
        break
    
print(times)

#creating a csv file with the times an object entered and left the frame.
for i in range(0,len(times),2):
    df = df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)

df.to_csv("Times.csv")
video.release()
cv2.destroyAllWindows